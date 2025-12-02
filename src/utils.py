import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MovieLens Recommender System')

    # ============ data ============
    parser.add_argument('--data_path', type=str, default='./data/ratings.dat')
    parser.add_argument('--num_features', type=int, default=64)

    # ============ model ============
    parser.add_argument('--model_type', type=str, default='gmf', choices=['mf', 'gmf', 'ncf'])

    # ============ train ============
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss_type', type=str, default='mse',choices=['mse', 'l1'])
    parser.add_argument('--seed', type=int, default=77)

    # ============ save ============
    parser.add_argument('--save_dir', type=str, default='./output')

    # ============ device ============
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'])

    # ============ parse =============
    args = parser.parse_args()

    return args

def save_args(args, save_dir):
    param_path = os.path.join(save_dir, 'config.json')
    with open(param_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Config saved to {os.path.abspath(param_path)}")

def load_args(save_dir):
    param_path = os.path.join(save_dir, 'config.json')
    with open(param_path, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

