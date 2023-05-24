import torch
import numpy as np
from src.torchmimic.models import StandardLSTM
from src.torchmimic import IHMBenchmark
import random

seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

root_dir = r'..\data\mimic'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# training_dataset = IHMDataset(root_dir, train=True)
# val_dataset = IHMDataset(root_dir, train=False)

# logger
exp_name = "ihm Benchmark"
config = {
    "LR": 0.001,
    "Weight Decay": 0.0001
}

# model = StandardLSTM(
#     n_classes=2,
#     hidden_dim=256,
#     num_layers=1,
#     dropout_rate=0.3,
#     bidirectional=True,
# )

model = StandardLSTM(
    n_classes=2,
    hidden_dim=128,
    num_layers=1,
    dropout_rate=0.3,
    bidirectional=True,
)

trainer = IHMBenchmark(
    model=model,
    train_batch_size=8,
    test_batch_size=256,
    data=root_dir,
    learning_rate=0.001,
    weight_decay=0,
    report_freq=1000,
    device=device,
    workers=0,
    sample_size=None,
    wandb=False,
)

trainer.fit(100)
