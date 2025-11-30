import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import MovieLensDataset
from src.model import MatrixFactorization
from src.utils import parse_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def train(args):
    # save
    save_dir = './checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # data
    dataset = MovieLensDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # shuffle 一般采用硬编码

    # model
    model = MatrixFactorization(dataset.num_users, dataset.num_items, args.num_features)
    model = model.to(args.device)

    # optimization
    loss_choices = {
        'mse': nn.MSELoss(),
        'l1':  nn.L1Loss()
    }
    criterion = loss_choices[args.loss_type]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for user_ids, item_ids, ratings in dataloader:

            user_ids = user_ids.to(args.device)
            item_ids = item_ids.to(args.device)
            ratings = ratings.to(args.device)

            optimizer.zero_grad()

            output = model(user_ids, item_ids)

            loss = criterion(output, ratings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(dataloader)

        print(f'Epoch {epoch+1}/{args.num_epochs}: Average Loss = {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print(f"\tPreds: {output[:3].detach().cpu().numpy()}")
        print(f"\tTruth: {ratings[:3].detach().cpu().numpy()}")
        print("-" * 30)

    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))

def main():
    args = parse_args()
    set_seed(114514)
    train(args)

if __name__ == '__main__':
    main()