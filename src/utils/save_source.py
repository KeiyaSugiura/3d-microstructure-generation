import os
import shutil


def save_source(save_dir, ignore_prefix=".", ignore_name_list=["results"]) -> None:
    ignore = shutil.ignore_patterns("__pycache__")
    
    for name in os.listdir():
        if name in ignore_name_list:
            continue
        
        if name.startswith(ignore_prefix):
            continue
        
        if os.path.isdir(name):
            # フォルダごとコピー
            shutil.copytree(name, os.path.join(save_dir, name), ignore=ignore)
        else:
            shutil.copy(name, os.path.join(save_dir, name))
    
    shutil.make_archive(save_dir, format="zip", root_dir=save_dir)
    shutil.rmtree(save_dir)