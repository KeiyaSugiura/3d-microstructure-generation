import argparse
import gc
import os
import shutil

import torch
import torch.nn as nn
import yaml
from colorama import Fore, Style

import wandb
from data.get_dataloader import get_dataloader
from data.preprocessing import preprocessing
from models.get_model import get_critic, get_generator
from utils.calculate_gradient_penalty import calculate_gradient_penalty
from utils.get_optimizer import get_optimizer
from plots.plot2d_all import plot2d_all
from utils.save_source import save_source
from utils.seed_everything import seed_everything

# ターミナルの文字色を設定
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

def train(config, outdir):
    with wandb.init(config=config, project=config["project_name"], name=config["run_name"]):
        config = wandb.config
        
        xy_zx_yz_dataset = preprocessing(config)
        
        device = torch.device(f"cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")
        print(f'Using device: {device}')
        
        xy_dataloader = get_dataloader(config, xy_zx_yz_dataset[0])
        zx_dataloader = get_dataloader(config, xy_zx_yz_dataset[1])
        yz_dataloader = get_dataloader(config, xy_zx_yz_dataset[2])
        
        netG = get_generator(config).to(device)
        if ('cuda' in str(device)) and (config.ngpu > 1):
            netG = nn.DataParallel(netG, list(range(config.ngpu)))
        optG = get_optimizer(config, netG)
        
        netCs = []
        optCs = []
        for i in range(3):
            netC = get_critic(config).to(device)
            if ('cuda' in str(device)) and (config.ngpu > 1):
                netC = (nn.DataParallel(netC, list(range(config.ngpu)))).to(device)
            netCs.append(netC)
            optCs.append(get_optimizer(config, netC))
        
        print("Starting Training Loop...")
        plane_labels = ['xy', 'zx', 'yz']
        for epoch in range(1, config.num_epochs + 1):
            for iter, (xy_data, zx_data, yz_data) in enumerate(zip(xy_dataloader, zx_dataloader, yz_dataloader), 1):
                noise = torch.randn(config.batch_size, config.z_dim, config.z_size, config.z_size, config.z_size, device=device)
                fake3d = netG(noise).detach()
                for dim, (netC, optC, data, channel_dim_idx, height_dim_idx, width_dim_idx) in enumerate(zip(netCs, optCs, [xy_data, zx_data, yz_data], [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    netC.zero_grad()
                    real2d = data[0].to(device)
                    out_real = netC(real2d).view(-1).mean()
                    fake2d = fake3d.permute(0, channel_dim_idx, 1, height_dim_idx, width_dim_idx).reshape(config.crop_size * config.batch_size, config.num_phases, config.crop_size, config.crop_size)
                    out_fake = netC(fake2d).mean()
                    gradient_penalty = calculate_gradient_penalty(config, netC, real2d, fake2d[:config.batch_size], device)
                    errC = out_fake - out_real + gradient_penalty
                    errC.backward()
                    optC.step()
                    
                    wandb.log({
                        "epoch": epoch,
                        f"errC.{plane_labels[dim]}": errC,
                        f"errC.{plane_labels[dim]}-critic.output_real": out_real,
                        f"errC.{plane_labels[dim]}-critic.output_fake": out_fake,
                        f"errC.{plane_labels[dim]}-critic.wasserstein_distance": out_fake - out_real,
                        f"errC.{plane_labels[dim]}-critic.gradient_penalty": gradient_penalty
                    })
                    print(f'[{epoch}/{config.num_epochs}][{iter}/{len(xy_dataloader)}]\terrC: {errC.item():.4f}')
                
                if iter % int(config.critic_iters) == 0:
                    netG.zero_grad()
                    errG = 0
                    noise = torch.randn(config.batch_size, config.z_dim, config.z_size, config.z_size, config.z_size, device=device)
                    fake3d = netG(noise)
                    for dim, (netC, channel_dim_idx, height_dim_idx, width_dim_idx) in enumerate(zip(netCs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                        fake2d = fake3d.permute(0, channel_dim_idx, 1, height_dim_idx, width_dim_idx).reshape(config.crop_size * config.batch_size, config.num_phases, config.crop_size, config.crop_size)
                        out_fake = netC(fake2d).mean()
                        errG -= out_fake
                        
                        wandb.log({
                            f"errG.{plane_labels[dim]}-critic.output_fake": out_fake
                        })
                    errG.backward()
                    optG.step()
                    
                    wandb.log({
                        "errG": errG
                    })
                    print(f'{y_}[{epoch}/{config.num_epochs}][{iter}/{len(xy_dataloader)}]\terrG: {errG.item():.4f}{sr_}')
                
                if iter % 25 == 0:
                    netG.eval()
                    with torch.no_grad():
                        os.makedirs(outdir + 'generators/', exist_ok=True)
                        os.makedirs(outdir + 'critics/', exist_ok=True)
                        torch.save(netG.state_dict(), outdir + f'generators/{config.run_name}_generator_epoch{epoch}_iter{iter}.pt')
                        torch.save(netC.state_dict(), outdir + f'critics/{config.run_name}_critic_epoch{epoch}_iter{iter}.pt')
                        noise = torch.randn(1, config.z_dim, config.z_size, config.z_size, config.z_size, device=device)
                        fake3d = netG(noise)
                        plot2d_all(config, epoch, iter, fake3d, 5, outdir)
                    netG.train()
                
                del noise, fake3d, fake2d, out_real, out_fake, errC
                gc.collect()
                torch.cuda.empty_cache()
                
                if iter % int(config.critic_iters) == 0:
                    del errG
                    gc.collect()
                    torch.cuda.empty_cache()


if "__main__" == __name__:
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/debug.yml")
    args = parser.parse_args()
    config_path = args.config
    
    # 設定ファイル
    config = None
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None, f"設定ファイルが正しく読み込めませんでした: {config_path}"
    
    # 結果フォルダ
    outdir = f"../output/{config['material_name']}/{config['run_name']}/"
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(config_path, f"{outdir}/{os.path.basename(config_path)}")
    
    # 実行時のソースコード一式を zip 圧縮して結果フォルダにコピー(バックアップ)
    source_dir = os.path.join(outdir, '_src_backup')
    os.makedirs(source_dir)
    save_source(source_dir)
    
    # 乱数シードの固定
    seed_everything(config)
    
    # 学習
    train(config, outdir)