import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
from torchdiffeq import odeint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger



class ODEFunc(nn.Module):
    def __init__(self, M, K, A, layer_sizes, activation, init_method, dropout_rate=0.0):
        super(ODEFunc, self).__init__()
        self.M = M
        self.K = K
        self.A = A
        layers = []
        input_dim = self.M + self.A + 1 + self.K  # x(t) and t and v
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        output_dim = self.M + self.A
        layers.append(nn.Linear(input_dim, output_dim))  # Output layer
        self.net = nn.Sequential(*layers)
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init_method(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, xav):
        # xv: tensor of shape [batch_size, M+A+K]
        # - first M = x(t)
        #  - next A = a(t)
        #  - last K = v (static) 
        # t: scalar tensor
        # We output [dx/dt, da/dt, dv/dt=0].
        batch_size = xav.shape[0]
        t_expanded = t.expand(batch_size, 1)  # [batch_size, 1]
        x = xav[:, :self.M]  # [batch_size, M]
        a = xav[:, self.M:self.M + self.A]  # [batch_size, A]
        v = xav[:, self.M+self.A:]  # [batch_size, K]
        # Combine x, t, and v
        input_data = torch.cat([x, a, t_expanded, v], dim=-1)
        dx_da = self.net(input_data)
        dx = dx_da[:, :self.M]  # [batch_size, M]
        da = dx_da[:, self.M:]  # [batch_size, A]
        # dv/dt = 0
        dv = torch.zeros_like(v)
        return torch.cat([dx, da, dv], dim=-1)  # [batch_size, M+A+K]

class NeuralODERegressor(pl.LightningModule):
    def __init__(
        self,
        M,
        K,
        layer_sizes=[16],
        activation=nn.Tanh,
        init_method=nn.init.xavier_normal_,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout_rate=0.0,
        solver='rk4',
        solver_options=None,
        device='cpu',
        seed=0,
        augment_dim=0,
    ):
        super(NeuralODERegressor, self).__init__()
        
        # Set seed for reproducibility
        self.seed = seed
        self._set_seed()
        self.M = M
        self.K = K
        self.A = augment_dim

        self.save_hyperparameters() # this will make the arguments accessible through self.hparams
        self.func = ODEFunc(self.M, self.K, self.A, layer_sizes, activation, init_method, dropout_rate)
        self.loss_fn = nn.MSELoss()

        # Data normalization parameters
        self.y_mean = None
        self.y_std = None
        self.t_mean = None
        self.t_std = None
        self.v_mean = None
        self.v_std = None

        self.ts_norm = None



    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        pl.seed_everything(self.seed, workers=True)

    def forward(self, x0_batch, v_batch, t):
        # x0_batch: tensor of shape [batch_size, M]
        # v_batch: tensor of shape [batch_size, K]
        # t: tensor of shape [sequence_length]

        batch_size = x0_batch.shape[0]

        if self.A > 0:
            # Augment x0_batch with zeros
            a0_batch = torch.zeros(batch_size, self.A).to(x0_batch.device)
            xav0_batch = torch.cat([x0_batch, a0_batch, v_batch], dim=-1)
        else:
            xav0_batch = torch.cat([x0_batch, v_batch], dim=-1)


        pred_xv = odeint(
            self.func,
            xav0_batch,
            t,
            method=self.hparams.solver,
            options=self.hparams.solver_options,
        )
        # pred_xv: [sequence_length, batch_size, M+A+K]
        pred_x = pred_xv[:, :, :self.M]  # [sequence_length, batch_size, M]
        pred_x = pred_x.permute(1, 0, 2)  # [batch_size, sequence_length, M]
        return pred_x

    def training_step(self, batch, batch_idx):
        x0_batch, y_true_batch, v_batch = batch
        ts = self.ts_norm.to(self.device)  # [sequence_length]
        pred_y_batch = self.forward(x0_batch, v_batch, ts)
        loss = self.loss_fn(pred_y_batch, y_true_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x0_batch, y_true_batch, v_batch = batch
        ts = self.ts_norm.to(self.device)
        pred_y_batch = self.forward(x0_batch, v_batch, ts)
        loss = self.loss_fn(pred_y_batch, y_true_batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.func.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def _normalize_data(self, xs, ys, ts, vs):

        self.y_mean = ys.float().mean()
        self.y_std = ys.float().std()
        self.t_mean = ts.float().mean()
        self.t_std = ts.float().std()
        self.v_mean = vs.float().mean()
        self.v_std = vs.float().std()

    def fit(self, train_xs, train_ts, train_ys, train_vs, val_xs, val_ts, val_ys, val_vs, batch_size=32, max_epochs=100, tuning=False):
        """
        Fits the Neural ODE model to the provided training data with early stopping on validation data.

        Parameters:
        - train_xs (numpy array): Initial conditions for training trajectories, shape [num_train_samples, M]
        - train_ts (numpy array): Time points for training trajectories, shape [sequence_length]
        - train_ys (numpy array): Observed training trajectories, shape [num_train_samples, sequence_length, M]
        - train_vs (numpy array): Observed training static features, shape [num_train_samples, K]
        - val_xs (numpy array): Initial conditions for validation trajectories, shape [num_val_samples, M]
        - val_ts (numpy array): Time points for validation trajectories, shape [sequence_length]
        - val_ys (numpy array): Observed validation trajectories, shape [num_val_samples, sequence_length, M]
        - val_vs (numpy array): Observed validation static features, shape [num_val_samples, K]
        - batch_size (int): Number of samples per batch.
        - max_epochs (int): Number of training epochs.
        """
        # Convert numpy arrays to torch tensors
        train_xs = torch.tensor(train_xs)
        train_ts = torch.tensor(train_ts)
        train_ys = torch.tensor(train_ys)
        train_vs = torch.tensor(train_vs)
        val_xs = torch.tensor(val_xs)
        val_ts = torch.tensor(val_ts)
        val_ys = torch.tensor(val_ys)
        val_vs = torch.tensor(val_vs)

        # Ensure time steps are the same for training and validation
        if not torch.allclose(train_ts, val_ts):
            raise ValueError("Time steps for training and validation sets must be the same.")

        # Normalize data
        self._normalize_data(train_xs, train_ys, train_ts, train_vs)

        # Normalize training data
        train_xs_norm = ((train_xs - self.y_mean) / self.y_std).float()
        train_ys_norm = ((train_ys - self.y_mean) / self.y_std).float()
        self.ts_norm = ((train_ts - self.t_mean) / self.t_std).float()  # [sequence_length]
        train_vs_norm = ((train_vs - self.v_mean) / self.v_std).float()

        # Create training dataset and loader
        train_dataset = TensorDataset(train_xs_norm, train_ys_norm, train_vs_norm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Normalize validation data
        val_xs_norm = ((val_xs - self.y_mean) / self.y_std).float()
        val_ys_norm = ((val_ys - self.y_mean) / self.y_std).float()
        val_vs_norm = ((val_vs - self.v_mean) / self.v_std).float()

        # Create validation dataset and loader
        val_dataset = TensorDataset(val_xs_norm, val_ys_norm, val_vs_norm)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Set up early stopping and model checkpointing
        early_stopping = EarlyStopping('val_loss',
                                       patience=10 if tuning else 20,
                                        mode='min')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='best_model',
            save_top_k=1,
            mode='min'
        )

        # Determine accelerator and devices based on the specified device
        if self.hparams.device == 'cpu':
            accelerator = 'cpu'
            devices = 1
        elif self.hparams.device in ['gpu', 'cuda']:
            accelerator = 'gpu'
            devices = 1  # Adjust this if you want to use multiple GPUs
        else:
            raise ValueError(f"Unsupported device type: {self.hparams.device}")
        
        wandb_logger = WandbLogger(project="PersonalizedSemanticODE",save_dir="lightning_logs")

        # Initialize the Trainer with the specified device
        trainer = pl.Trainer(
            max_epochs = max_epochs//2 if tuning else max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=wandb_logger,
            enable_checkpointing=True,
            accelerator=accelerator,
            devices=devices,
            deterministic=True,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            enable_progress_bar=True
        )

        # Fit the model
        trainer.fit(self, train_loader, val_loader)

        # Load the best model
        self.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])

    def predict(self, xs, ts, vs):
        """
        Predicts trajectories based on initial conditions and time points.

        Parameters:
        - xs (numpy array): Initial conditions, shape [num_samples, M]
        - ts (numpy array): Time points, shape [sequence_length] or [num_samples, sequence_length]
        - vs (numpy array): Static features, shape [num_samples, K]

        Returns:
        - preds (numpy array): Predicted trajectories, shape [num_samples, sequence_length]
        """
        # Convert numpy arrays to torch tensors
        xs_tensor = torch.tensor(xs)
        xs_tensor = xs_tensor.to(self.device)
        ts_tensor = torch.tensor(ts)
        ts_tensor = ts_tensor.to(self.device)
        vs_tensor = torch.tensor(vs)
        vs_tensor = vs_tensor.to(self.device)

        xs_tensor = ((xs_tensor - self.y_mean) / self.y_std).float()
        vs_tensor = ((vs_tensor - self.v_mean) / self.v_std).float()

        # Check if ts is the same for all samples
        if ts_tensor.ndim == 1:
            # ts is shared across all samples
            self.eval()
            with torch.no_grad():
                ts_norm = ((ts_tensor - self.t_mean) / self.t_std).float()
                pred_y = self.forward(xs_tensor.to(self.device), vs_tensor.to(self.device), ts_norm.to(self.device))
                pred_y_denorm = pred_y * self.y_std + self.y_mean
                return pred_y_denorm.cpu().numpy()
        else:
            # ts differs per sample
            preds = []
            self.eval()
            with torch.no_grad():
                for x0, t in zip(xs_tensor, ts_tensor, vs_tensor):
                    x0 = x0.unsqueeze(0)  # [1, M]
                    v = v.unsqueeze(0) # [1, K]
                    t_norm = ((t - self.t_mean) / self.t_std).float()
                    pred_y = self.forward(x0.to(self.device), v.to(self.device), t_norm.to(self.device))
                    # pred_y = odeint(
                    #     self.func,
                    #     x0.to(self.device),
                    #     t_norm.to(self.device),
                    #     method=self.hparams.solver,
                    #     options=self.hparams.solver_options,
                    # )
                    # pred_y = pred_y.squeeze(-1).squeeze(1)  # [sequence_length]
                    pred_y_denorm = pred_y * self.y_std + self.y_mean
                    preds.append(pred_y_denorm.cpu())
            preds = torch.cat(preds, axis=0).numpy()
            return preds
