import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dataset import MovieLensDataset
from src import model
from src.utils import parse_args, load_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(input_args):
    # Configure device
    device = input_args.device
    print(f"Using device: {device}")

    # Confirm save_dir
    save_dir = input_args.save_dir
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Experiment directory not found: {save_dir}")

    # Load train_args
    train_args = load_args(save_dir)

    # Load train seed
    set_seed(train_args.seed)

    # Load dataset
    dataset = MovieLensDataset(train_args.data_path)

    # Recreate dataset
    data_size = len(dataset)
    train_size = int(train_args.train_ratio * data_size)
    test_size = data_size - train_size

    _, test_set = random_split(dataset, [train_size, test_size])

    # Create dataloader
    test_loader = DataLoader(test_set, batch_size=train_args.batch_size, shuffle=False)

    # Select model
    model_mapping = {
        'mf': model.MatrixFactorization,
        'gmf': model.GeneralizedMF,
        'ncf': model.NeuralCF,
    }
    if train_args.model_type not in model_mapping:
        raise ValueError(f"Model '{train_args.model_type}' not found. Choices: {list(model_mapping.keys())}")
    model_class = model_mapping[train_args.model_type]

    # Initialize model architecture
    net = model_class(dataset.num_users, dataset.num_items, train_args.num_features)
    net = net.to(device)

    # Load trained model weights
    model_path = os.path.join(save_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found: {model_path}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded weights from: {os.path.abspath(model_path)}")

    # Start inference
    net.eval()
    all_preds = []
    all_ratings = []

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            output = net(user_ids, item_ids)

            all_preds.extend(output.cpu().numpy())
            all_ratings.extend(ratings.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_ratings)

    # Clipping to [1, 5]
    y_pred = np.clip(y_pred, 1.0, 5.0)

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(" Evaluation Results ".center(60, "="))
    print(f"Model: {train_args.model_type.upper()}")
    print(f"Seed:  {train_args.seed} (Restored)")
    print(f"Test Set Size: {len(y_true)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("=" * 60)

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()