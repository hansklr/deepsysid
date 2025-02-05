import json
import logging
import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from numpy.typing import NDArray

from deepsysid.models.adversarial.adversaries.standard_adversaries import PgdRR
from deepsysid.models.adversarial.datasets_robust import RobustPredictorDataset, SymmetricRobustPredictorDataset
from deepsysid.networks import loss, rnn
from deepsysid.models import utils
from deepsysid.models.base import DynamicIdentificationModelConfig, DynamicIdentificationModel
from deepsysid.models.datasets import RecurrentInitializerDataset

logger = logging.getLogger(__name__)


class RobustLSTMInitModelConfig(DynamicIdentificationModelConfig):
    pgd_rr: PgdRR
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float
    train_initializer_sequence_length: int
    initializer_sequence_length: int
    predictor_sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    loss: Literal['mse', 'msge']

    class Config:
        arbitrary_types_allowed = True


class RobustLSTMInitModel(DynamicIdentificationModel):
    CONFIG = RobustLSTMInitModelConfig

    def __init__(self, config: RobustLSTMInitModelConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.pgd = config.pgd_rr

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers = config.num_recurrent_layers
        self.dropout = config.dropout

        self.train_initializer_sequence_length = config.train_initializer_sequence_length
        self.initializer_sequence_length = config.initializer_sequence_length
        self.predictor_sequence_length = config.predictor_sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor = config.epochs_predictor

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self.predictor = rnn.BasicLSTM(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self.predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self.initializer.parameters(), lr=self.learning_rate
        )

        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_std: Optional[NDArray[np.float64]] = None
        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]]
    ) -> Dict[str, NDArray[np.float64]]:
        epoch_losses_initializer = []
        epoch_losses_predictor = []

        self.predictor.train()
        self.initializer.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]

        initializer_dataset = RecurrentInitializerDataset(
            control_seqs, state_seqs, self.train_initializer_sequence_length
        )

        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            print(i)
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y = self.initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                self.optimizer_init.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer} '
                f'- Epoch Loss (Initializer): {total_loss}'
            )
            epoch_losses_initializer.append([i, total_loss])

        time_end_init = time.time()
        robust_predictor_dataset = data.Dataset
        if self.initializer_sequence_length == self.predictor_sequence_length:
            robust_predictor_dataset = SymmetricRobustPredictorDataset(
                self.pgd,
                control_seqs,
                state_seqs,
                self.initializer_sequence_length
            )
        else:
            robust_predictor_dataset = RobustPredictorDataset(
                self.pgd,
                control_seqs,
                state_seqs,
                self.initializer_sequence_length,
                self.predictor_sequence_length
            )
        time_start_pred = time.time()
        for i in range(self.epochs_predictor):
            print(i)
            data_loader = data.DataLoader(
                robust_predictor_dataset, self.batch_size, shuffle=True, drop_last=False
            )
            total_loss = 0
            for batch_idx, batch in enumerate(data_loader):
                self.predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self.initializer.forward(
                    batch['x0'].float().to(self.device), return_state=True
                )
                # Predict and optimize
                y = self.predictor.forward(batch['x'].float().to(self.device), hx=hx)
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_pred.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor} '
                f'- Epoch Loss (Predictor): {total_loss}'
            )
            epoch_losses_predictor.append([i, total_loss])

        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        return dict(
            epoch_loss_initializer=np.array(epoch_losses_initializer, dtype=np.float64),
            epoch_loss_predictor=np.array(epoch_losses_predictor, dtype=np.float64),
            training_time_initializer=np.array([time_total_init], dtype=np.float64),
            training_time_predictor=np.array([time_total_pred], dtype=np.float64),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self.initializer.eval()
        self.predictor.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            _, hx = self.initializer.forward(init_x, return_state=True)
            y = self.predictor.forward(pred_x, hx=hx, return_state=False)
            # We do this just to get proper type hints.
            # Option 1 should always execute until we change the signature.
            if isinstance(y, torch.Tensor):
                y_np: NDArray[np.float64] = (
                    y.cpu().detach().squeeze().numpy().astype(np.float64)
                )
            else:
                y_np = y[0].cpu().detach().squeeze().numpy().astype(np.float64)

        y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.initializer.state_dict(), file_path[0])
        torch.save(self.predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'state_mean': self.state_mean.tolist(),
                    'state_std': self.state_std.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self.initializer.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)  # type: ignore
        )
        self.predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)  # type: ignore
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self.state_std = np.array(norm['state_std'], dtype=np.float64)
        self.control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self.control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count
