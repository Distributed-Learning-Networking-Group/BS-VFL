import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from src.torchmimic.loggers import IHMLogger
from src.torchmimic.data import IHMDataset
from src.torchmimic.utils import pad_colalte


class IHMBenchmark:
    def __init__(
        self,
        model,
        train_batch_size=8,
        test_batch_size=256,
        data="/data/datasets/mimic3-benchmarks/data/in-hospital-mortality/",
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        exp_name="Test",
        device="cpu",
        sample_size=None,
        workers=5,
        wandb=False,
    ):

        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.device = device
        self.report_freq = report_freq

        config = {}
        config.update(model.get_config())  # hidden_dim num_layers dropout_rate bidirectional
        config.update(self.get_config())  # test_batch_size train_batch_size learning_rate weight_decay

        self.logger = IHMLogger(exp_name + "_ihm", config, wandb)

        torch.cuda.set_device(self.device)

        train_dataset = IHMDataset(
            data,
            train=True,
            n_samples=sample_size,
        )

        test_dataset = IHMDataset(
            data,
            train=False,
            n_samples=sample_size,
        )

        kwargs = {"num_workers": workers, "pin_memory": True} if self.device else {}

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=pad_colalte,
            **kwargs,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        self.model = self.model.to(self.device)

        # self.optimizer = optim.Adam(
        #     model.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        #     betas=(0.9, 0.98),
        # )
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # self.crit = nn.BCELoss()
        self.crit = nn.CrossEntropyLoss()

    def fit(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            self.logger.reset()
            for batch_idx, (data, label, lens, mask) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model((data, lens))
                # label_one_hot = torch.zeros_like(output).scatter_(1, label[:, None].to(torch.int64), 1)
                # loss = self.crit(output, label[:, None])
                loss = self.crit(output, label.long())
                # print(data[0][0])
                # print(mask[0][0])
                # exit()

                # softmax = nn.Softmax(dim=1)
                # output_prob = softmax(output)

                # loss = self.crit(output, label_one_hot)
                loss.backward()

                # print('output:', output.size())
                # print('label:', label_one_hot)
                # exit()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.logger.update(output, label, loss)
                if (batch_idx + 1) % self.report_freq == 0:
                    print(f"Train: epoch: {epoch+1}, loss = {self.logger.get_loss()}")
            self.logger.print_metrics(epoch, split="Train")

            self.model.eval()
            self.logger.reset()
            with torch.no_grad():
                for batch_idx, (data, label, lens, mask) in enumerate(self.test_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)

                    output = self.model((data, lens))
                    # label_one_hot = torch.zeros_like(output).scatter_(1, label[:, None].to(torch.int64), 1)
                    # loss = self.crit(output, label[:, None])
                    loss = self.crit(output, label.long())
                    # softmax = nn.Softmax(dim=1)
                    # output_prob = softmax(output)
                    #
                    # loss = self.crit(output, label_one_hot)
                    self.logger.update(output, label, loss)

                    if (batch_idx + 1) % self.report_freq == 0:
                        print(
                            f"Eval: epoch: {epoch+1}, loss = {self.logger.get_loss()}"
                        )

                self.logger.print_metrics(epoch, split="Eval")


    def get_config(self):
        return {
            "test_batch_size": self.test_batch_size,
            "train_batch_size": self.train_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
