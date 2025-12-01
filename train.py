import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.dataset import MovieLensDataset
from src import model
from src.utils import parse_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def train(args):
    # Setup save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load dataset
    dataset = MovieLensDataset(args.data_path)

    # Split dataset into train_set & test_set
    data_size = len(dataset)
    train_size = int(args.train_ratio * data_size)
    test_size = data_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model_mapping = {
        'mf': model.MatrixFactorization,
        'gmf': model.GeneralizedMF,
        'ncf': model.NeuralCF,
    }
    if args.model_type not in model_mapping:
        raise ValueError(f"Model '{args.model_type}' not found. Choices: {list(model_mapping.keys())}")

    model_class = model_mapping[args.model_type]

    net = model_class(dataset.num_users, dataset.num_items, args.num_features)

    net = net.to(args.device)

    # Define loss func & optimizer
    loss_choices = {
        'mse': nn.MSELoss(),
        'l1':  nn.L1Loss()
    }
    criterion = loss_choices[args.loss_type]
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Initialize best_loss
    best_loss = float('inf')

    # Start training loop
    for epoch in range(args.num_epochs):

        # ============ Train ============
        net.train()
        total_train_loss = 0

        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(args.device)
            item_ids = item_ids.to(args.device)
            ratings = ratings.to(args.device)

            optimizer.zero_grad()

            # Forward
            output = net(user_ids, item_ids)

            # Compute loss
            loss = criterion(output, ratings)

            # Backward & Optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ============ Eval ============
        net.eval()
        total_test_loss = 0

        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(args.device)
                item_ids = item_ids.to(args.device)
                ratings = ratings.to(args.device)

                output = net(user_ids, item_ids)
                loss = criterion(output, ratings)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)

        # ============ Log & Save ============
        print(f'Epoch {epoch+1}/{args.num_epochs}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}')

        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

        print(f"\tPreds: {output[:3].detach().cpu().numpy()}")
        print(f"\tTruth: {ratings[:3].detach().cpu().numpy()}")
        print("-" * 30)

    # Save final model
    torch.save(net.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))
    print(f"Training Done! Best Loss: {best_loss:.4f}")

def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)

if __name__ == '__main__':
    main()