import argparse

import numpy as np
import tifffile
import torch
import yaml

from data.postprocessing import postprocessing
from models.get_model import get_generator
from utils.seed_everything import seed_everything


def infer(config, z_size, outdir, epoch, iter):
    netG = get_generator(config)
    netG.load_state_dict(torch.load(outdir + f'generators/{config["run_name"]}_generator_epoch{epoch}_iter{iter}.pt'))
    
    netG.eval()
    noise = torch.randn(1, config["z_dim"], z_size, z_size, z_size)
    with torch.no_grad():
        fake_3d = netG(noise)
    
    img = postprocessing(fake_3d)[0]
    img = np.int_(img)
    tifffile.imwrite(outdir + f'{config["run_name"]}_epoch{epoch}_iter{iter}.tif', img)


if "__main__" == __name__:
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/debug.yml")
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-i", "--iter", type=int, default=100)
    args = parser.parse_args()
    config_path = args.config
    
    # 設定ファイル
    config = None
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None, f"設定ファイルが正しく読み込めませんでした: {config_path}"
    
    # 結果フォルダ
    outdir = f"../output/{config['material_name']}/{config['run_name']}/"
    
    # 乱数シードの固定
    seed_everything(config)
    
    # 推論
    infer(config, 6, outdir, epoch=args.epoch, iter=args.iter)