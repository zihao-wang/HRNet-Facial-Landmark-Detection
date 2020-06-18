import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default='data')

def rm_unused_files(dir, filter):
    sub_list = [os.path.join(dir, f) for f in os.listdir(dir)]
    for s in sub_list:
        if os.path.isdir(s):
            rm_unused_files(s, filter)
        elif filter(s):
            print("remove ", s)
            os.remove(s)

def filter(s):
    filename = s.split('/')[-1]
    if "DS_Store" in filename:
        return True
    if "ipynb_checkpoints" in filename:
        return True
    if '._' in filename:
        return True
    return False

if __name__=="__main__":
    args = parser.parse_args()
    rm_unused_files(args.dir, filter)
