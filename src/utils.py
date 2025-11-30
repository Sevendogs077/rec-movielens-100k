import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='MovieLens 推荐系统训练参数配置')

    # data
    parser.add_argument('--data_path', type=str, default='./data/u.data')
    parser.add_argument('--num_features', type=int, default=32)

    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--loss_type', type=str, default='mse',choices=['mse', 'l1'])

    # device
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    args.device = torch.device('cuda') if args.device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    return args



