import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    #print(args.offset)
    #print(args.n_parts)

    n_parts = 4
    offset = 0
    for gpu in range(4):
        cmd = f'CUDA_VISIBLE_DEVICES={gpu} python3 /opt/ml/generate.py --n_parts {n_parts} --part {gpu+offset}'
        print(cmd)
        os.system(cmd)
