import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
from torchdiffeq import odeint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


class LatentODEFunc(nn.Module):
    """
    ODE function for z(t) in a Latent ODE model.
    The dynamic state here is z(t) in R^L, plus v in R^K (static).
    So total dimension is L + K. We also pass time t.
    """
    def __init__(self, latent_dim, K, layer_sizes, activation, init_method, dropout_rate=0.0):
        super(LatentODEFunc, self).__init__()
        self.latent_dim = latent_dim

        # Build a simple MLP for f(z, v, t) -> dz/dt
        input_dim = latent_dim  + 1  # z(t), and time
        layers = []
        
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output is the derivative of z(t) => dimension = latent_dim
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init_method(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        """
        t is a scalar tensor
        z: [batch_size, L]
        """
        batch_size = z.shape[0]
        t_expanded = t.view(1, 1).expand(batch_size, 1)
        
        # Combine z(t) and t
        net_input = torch.cat([z, t_expanded], dim=1)
        
        dzdt = self.net(net_input)  # shape [batch_size, latent_dim]
        
        return dzdt  # shape [batch_size, L]


class Encoder(nn.Module):
    """
    Minimal encoder that maps [x(0), v] -> z(0).
    If you have full sequences, you could use an RNN or aggregator.
    """
    def __init__(self, M, K, latent_dim, layer_sizes, activation, init_method, dropout_rate=0.0):
        super(Encoder, self).__init__()
        input_dim = M + K  # from x(0) in R^M and v in R^K
        layers = []
        
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output dimension = latent_dim
        layers.append(nn.Linear(input_dim, latent_dim))
        self.net = nn.Sequential(*layers)

        # Init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init_method(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x0, v):
        """
        x0: [batch_size, M]
        v:  [batch_size, K]
        return z0: [batch_size, latent_dim]
        """
        inp = torch.cat([x0, v], dim=-1)
        z0 = self.net(inp)
        return z0


class Decoder(nn.Module):
    """
    Decoder that maps z(t) -> x(t).
    """
    def __init__(self, latent_dim, M, layer_sizes, activation, init_method, dropout_rate=0.0):
        super(Decoder, self).__init__()
        input_dim = latent_dim  # we decode from z(t)
        layers = []
        
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output dimension = M
        layers.append(nn.Linear(input_dim, M))
        self.net = nn.Sequential(*layers)

        # Init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init_method(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, z):
        """
        z: [batch_size, latent_dim]
        return x: [batch_size, M]
        """
        x = self.net(z)
        return x


class LatentODERegressor(pl.LightningModule):
    def __init__(
        self,
        M,
        K,
        latent_dim=8,          # dimension of the latent state z(t)
        encoder_sizes=[16],    # layer sizes for the encoder
        decoder_sizes=[16],    # layer sizes for the decoder
        odefunc_sizes=[16],    # layer sizes for the ODE function
        activation=nn.Tanh,
        init_method=nn.init.xavier_normal_,
        learning_rate=1e-3,
        weight_decay=0.0,
        dropout_rate=0.0,
        solver='rk4',
        solver_options=None,
        device='cpu',
        seed=0
    ):
        super().__init__()
        pl.seed_everything(seed, workers=True)
        
        self.M = M
        self.K = K
        self.latent_dim = latent_dim
        
        self.save_hyperparameters()  # saves all init arguments to self.hparams

        # Build networks
        self.encoder = Encoder(M, K, latent_dim, encoder_sizes, activation, init_method, dropout_rate)
        self.odefunc = LatentODEFunc(latent_dim, K, odefunc_sizes, activation, init_method, dropout_rate)
        self.decoder = Decoder(latent_dim, M, decoder_sizes, activation, init_method, dropout_rate)
        
        self.loss_fn = nn.MSELoss()
        
        # Data normalization parameters
        self.y_mean = None
        self.y_std = None
        self.t_mean = None
        self.t_std = None
        self.v_mean = None
        self.v_std = None
        self.ts_norm = None

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.solver = solver
        self.solver_options = solver_options


    def forward(self, x0_batch, v_batch, t):
        """
        1) Encode [x0, v] -> z0
        2) Integrate z(t) over time
        3) Decode z(t) to x(t)
        Return predicted x(t).
        
        x0_batch: [batch_size, M]
        v_batch:  [batch_size, K]
        t:        [sequence_length] (already normalized)
        """
        # 1) Encode
        z0 = self.encoder(x0_batch, v_batch)  # [batch_size, latent_dim]
        
        # ODE solve
        z_traj = odeint(
            self.odefunc,
            z0,       # initial state
            t,         # time steps
            method=self.solver,
            options=self.solver_options
        )
        # shape of zv_traj: [sequence_length, batch_size, L]
        

        # 2) Decode each z(t) => x(t)
        # We'll decode in a loop or vectorized. Let's do a simple vectorized approach:
        # z_traj = [sequence_length, batch_size, latent_dim]
        seq_len, batch_size, _ = z_traj.shape
        
        # We can reshape for decoder: => [sequence_length*batch_size, latent_dim]
        z_traj_reshaped = z_traj.reshape(seq_len * batch_size, self.latent_dim)
        
        # decode
        x_traj_reshaped = self.decoder(z_traj_reshaped)  # => [sequence_length*batch_size, M]
        
        # reshape back to => [sequence_length, batch_size, M]
        x_traj = x_traj_reshaped.reshape(seq_len, batch_size, self.M)
        
        # reorder to => [batch_size, sequence_length, M]
        x_traj = x_traj.permute(1, 0, 2)
        
        return x_traj

    def training_step(self, batch, batch_idx):
        x0_batch, y_true_batch, v_batch = batch
        # self.ts_norm: [sequence_length]
        ts = self.ts_norm.to(self.device)
        
        pred_y_batch = self.forward(x0_batch, v_batch, ts)  # [batch_size, sequence_length, M]
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
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.odefunc.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    # ---------------------------
    # Data normalization methods
    # ---------------------------
    def _normalize_data(self, xs, ys, ts, vs):
        self.y_mean = ys.float().mean()
        self.y_std = ys.float().std()
        self.t_mean = ts.float().mean()
        self.t_std = ts.float().std()
        self.v_mean = vs.float().mean()
        self.v_std = vs.float().std()

    def fit(self, 
            train_xs, train_ts, train_ys, train_vs,
            val_xs, val_ts, val_ys, val_vs,
            batch_size=32,
            max_epochs=100,
            tuning=False):
        """
        train_xs: shape [num_train_samples, M]  -> x(0)
        train_ts: shape [sequence_length]
        train_ys: shape [num_train_samples, sequence_length, M]
        train_vs: shape [num_train_samples, K]
        ...
        """
        # Convert numpy arrays to torch
        train_xs = torch.tensor(train_xs)
        train_ts = torch.tensor(train_ts)
        train_ys = torch.tensor(train_ys)
        train_vs = torch.tensor(train_vs)
        val_xs = torch.tensor(val_xs)
        val_ts = torch.tensor(val_ts)
        val_ys = torch.tensor(val_ys)
        val_vs = torch.tensor(val_vs)

        # Check that time steps match
        if not torch.allclose(train_ts, val_ts):
            raise ValueError("Time steps for training and validation must be the same.")

        # Normalize
        self._normalize_data(train_xs, train_ys, train_ts, train_vs)
        
        # Training set
        train_xs_norm = ((train_xs - self.y_mean) / self.y_std).float()
        train_ys_norm = ((train_ys - self.y_mean) / self.y_std).float()
        self.ts_norm = ((train_ts - self.t_mean) / self.t_std).float()
        train_vs_norm = ((train_vs - self.v_mean) / self.v_std).float()
        
        train_dataset = TensorDataset(train_xs_norm, train_ys_norm, train_vs_norm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation set
        val_xs_norm = ((val_xs - self.y_mean) / self.y_std).float()
        val_ys_norm = ((val_ys - self.y_mean) / self.y_std).float()
        val_vs_norm = ((val_vs - self.v_mean) / self.v_std).float()
        
        val_dataset = TensorDataset(val_xs_norm, val_ys_norm, val_vs_norm)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Callbacks
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

        # Device logic
        if self.hparams.device == 'cpu':
            accelerator = 'cpu'
            devices = 1
        elif self.hparams.device in ['gpu', 'cuda']:
            accelerator = 'gpu'
            devices = 1
        else:
            raise ValueError(f"Unsupported device type: {self.hparams.device}")

        wandb_logger = WandbLogger(project="LatentODEProject", save_dir="lightning_logs")

        trainer = pl.Trainer(
            max_epochs=max_epochs // 2 if tuning else max_epochs,
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
        
        trainer.fit(self, train_loader, val_loader)
        self.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])

    def predict(self, xs, ts, vs):
        """
        Predict full trajectories from initial conditions x(0), v.
        ts can be shape [sequence_length] or [num_samples, sequence_length].
        """
        xs_tensor = torch.tensor(xs, device=self.device)
        ts_tensor = torch.tensor(ts, device=self.device)
        vs_tensor = torch.tensor(vs, device=self.device)

        xs_tensor = ((xs_tensor - self.y_mean) / self.y_std).float()
        vs_tensor = ((vs_tensor - self.v_mean) / self.v_std).float()

        self.eval()
        with torch.no_grad():
            # Case: same time steps for all samples
            if ts_tensor.ndim == 1:
                ts_norm = ((ts_tensor - self.t_mean) / self.t_std).float()
                pred_y = self.forward(xs_tensor, vs_tensor, ts_norm)
                pred_y_denorm = pred_y * self.y_std + self.y_mean
                return pred_y_denorm.cpu().numpy()
            else:
                # Different time steps per sample
                preds = []
                for i in range(xs_tensor.shape[0]):
                    x0_i = xs_tensor[i].unsqueeze(0)
                    v_i = vs_tensor[i].unsqueeze(0)
                    t_i = ts_tensor[i]
                    t_norm = ((t_i - self.t_mean) / self.t_std).float()

                    pred_y_i = self.forward(x0_i, v_i, t_norm)
                    pred_y_i_denorm = pred_y_i * self.y_std + self.y_mean
                    preds.append(pred_y_i_denorm.cpu())

                return torch.cat(preds, dim=0).numpy()
