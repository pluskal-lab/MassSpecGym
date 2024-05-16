import torch
import torch.nn as nn
import pytorch_lightning as pl


# TODO: the main idea should be to have some callback for metrics which cannot be modified by subclasses
# This class should only implement metrics


class MassSpecGymModel(pl.LightningModule):
    
    def __init__(self, lr=1e-4, weight_decay=0):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.phi = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.phi(x)
        x = x.sum(dim=-1)
        x = self.rho(x)
        return x

    def step(self, x):
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, x)
        return loss, y_hat
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)[0]
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch)[0]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# class ExampleBaselineModel(MassSpecGymModel):

#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Linear(1, 1)

#     def forward(self, x):
#         return self.linear(x)

#     def step(self, x):
#         y_hat = self.forward(x)
#         loss = torch.nn.functional.mse_loss(y_hat, x)
#         return loss, y_hat
    
#     def training_step(self, batch, batch_idx):
#         return self.step(batch)[0]
    
#     def validation_step(self, batch, batch_idx):
#         return self.step(batch)[0]

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)