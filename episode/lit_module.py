import lightning as L
import torch
import os
from episode.torch_model import CubicModel

class LitSketchODE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.x0_included = config['x0_included']
        L.seed_everything(self.config['seed'])
        self.model = CubicModel(config)
        self.lr = self.config['lr']
        if 'refit' in self.config:
            self.refit = self.config['refit']
        else:
            self.refit = False
        self.dtw = self.config['dtw']
      

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.x0_included:
            X, B, T, X0, B_cat = batch
            B = (B, B_cat)
            return self.model.forward(X, B, T, X0=X0)
        else:
            X, B, T, B_cat = batch
            B = (B, B_cat)
            return self.model.forward(X,B,T)
    
    def forward(self, X, B, T, B_cat, X0=None):
        B = (B, B_cat)
        return self.model.forward(X, B, T, X0=X0)

    def training_step(self, batch, batch_idx):

        if self.refit:
            dtw = False
        else:
            dtw = self.dtw

        opt = self.optimizers()

        def closure():
            if self.x0_included:
                X, B, T, Y, X0, B_cat = batch
                B = (B, B_cat)
                opt.zero_grad()
                loss = self.model.loss(X, B, T, Y, X0=X0, with_derivative_loss=True, dtw=dtw)
            else:
                X, B, T, Y, B_cat = batch
                B = (B, B_cat)
                opt.zero_grad()
                loss = self.model.loss(X, B, T, Y, with_derivative_loss=True, dtw=dtw)
            
            # print(loss)
            self.manual_backward(loss)
            self.log('train_loss', loss)
            return loss

        opt.step(closure=closure)

    def validation_step(self, batch, batch_idx):

        if self.x0_included:
            X, B, T, Y, X0, B_cat = batch
            B = (B, B_cat)
            loss = self.model.loss(X, B, T, Y, X0=X0, with_derivative_loss=False)
        else:
            X, B, T, Y, B_cat = batch
            B = (B, B_cat)
            loss = self.model.loss(X, B, T, Y, with_derivative_loss=False)

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        if self.x0_included:
            X, B, T, Y, X0, B_cat = batch
            B = (B, B_cat)
            loss = self.model.loss(X, B, T, Y, X0=X0, with_derivative_loss=False)
        else:
            X, B, T, Y, B_cat = batch
            B = (B, B_cat)
            loss = self.model.loss(X, B, T, Y, with_derivative_loss=False)
        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr, history_size=100, max_iter=20, line_search_fn='strong_wolfe')
        # params=list()
        # params.extend(list(self.model.parameters()))
        # optimizer = LBFGSNew(self.model.parameters(), lr=self.lr, cost_use_gradient=True, history_size=100, max_iter=20)
        return optimizer