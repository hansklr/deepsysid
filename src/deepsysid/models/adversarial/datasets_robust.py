from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils import data

from deepsysid.models.adversarial.adversaries.standard_adversaries import PgdRR


class RobustPredictorDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        pgd_rr: PgdRR,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initializer_sequence_length: int,
        predictor_sequence_length: int,
    ):
        self.pgd_rr = pgd_rr
        self.initializer_sequence_length = initializer_sequence_length
        self.predictor_sequence_length = predictor_sequence_length
        self.total_sequence_length = initializer_sequence_length + predictor_sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x0, self.y0, self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        x0_seq = list()
        y0_seq = list()
        x_seq = list()
        x_adv_seq = list()
        y_seq = list()

        test_counter = 0

        for control, state in zip(control_seqs, state_seqs):
            test_counter += 1
            n_samples = int(
                control.shape[0] / self.total_sequence_length
            )

            x0 = np.zeros(
                (n_samples, self.initializer_sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            y0 = np.zeros((n_samples, self.state_dim), dtype=np.float64)
            x = np.zeros(
                (n_samples, self.predictor_sequence_length, self.control_dim), dtype=np.float64
            )
            y = np.zeros(
                (n_samples, self.predictor_sequence_length, self.state_dim), dtype=np.float64
            )
            x_adv = np.zeros(
                (n_samples, self.predictor_sequence_length, self.control_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                print(f'Sequence: {test_counter} of {len(control_seqs)}')
                print(f'Subsequence: {idx + 1} of {n_samples}')
                time = idx * self.total_sequence_length

                x0[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.initializer_sequence_length],
                        state[time : time + self.initializer_sequence_length, :],
                    )
                )
                y0[idx, :] = state[time + self.initializer_sequence_length - 1, :]
                x[idx, :, :] = control[
                    time + self.initializer_sequence_length : time + self.total_sequence_length, :
                ]
                y[idx, :, :] = state[
                    time + self.initializer_sequence_length : time + self.total_sequence_length, :
                ]

                # generate advex for x for half the sequences
                print(test_counter, len(control_seqs))
                if test_counter >= len(control_seqs) / 2:
                    current_pgd_rr = self.pgd_rr

                    current_x0 = torch.from_numpy(x0[idx, :, :]).unsqueeze(0).float()
                    current_x = torch.from_numpy(x[idx, :, :]).float()
                    current_y = torch.from_numpy(y[idx, :, :]).float()

                    _, hx = current_pgd_rr.lstm.initializer.forward(
                        current_x0.float().to('cpu'), return_state=True
                    )

                    advexs = current_pgd_rr.attack(current_x, current_y, hx)
                    strongest_advex = advexs[0][0]

                    x_adv[idx, :, :] = strongest_advex.detach().numpy()

            x0_seq.append(x0)
            y0_seq.append(y0)
            x_seq.append(x)
            y_seq.append(y)
            x_adv_seq.append(x)

            if test_counter >= len(control_seqs)/2:
                x0_seq.append(x0)
                y0_seq.append(y0)
                x_seq.append(x)
                y_seq.append(y)
                x_adv_seq.append(x_adv)

        return np.vstack(x0_seq), np.vstack(y0_seq), np.vstack(x_adv_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'x0': self.x0[idx],
            'y0': self.y0[idx],
            'x': self.x[idx],
            'y': self.y[idx],
        }


class SymmetricRobustPredictorDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
            self,
            pgd_rr: PgdRR,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
            sequence_length: int
    ):
        self.pgd_rr = pgd_rr
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x0, self.y0, self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        x0_seq = list()
        y0_seq = list()
        x_seq = list()
        x_adv_seq = list()
        y_seq = list()

        test_counter = 0
        for control, state in zip(control_seqs, state_seqs):
            test_counter += 1
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            x0 = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            y0 = np.zeros((n_samples, self.state_dim), dtype=np.float64)
            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )
            x_adv = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                print(f'Sequence: {test_counter} of {len(control_seqs)}')
                print(f'Subsequence: {idx + 1} of {n_samples}')
                time = idx * self.sequence_length

                x0[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.sequence_length],
                        state[time : time + self.sequence_length, :],
                    )
                )
                y0[idx, :] = state[time + self.sequence_length - 1, :]
                x[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                y[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                # generate advex for x
                if test_counter > len(control_seqs) / 2:
                    current_pgd_rr = self.pgd_rr

                    current_x0 = torch.from_numpy(x0[idx, :, :]).unsqueeze(0).float()
                    current_x = torch.from_numpy(x[idx, :, :]).float()
                    current_y = torch.from_numpy(y[idx, :, :]).float()

                    _, hx = current_pgd_rr.lstm.initializer.forward(
                        current_x0.float().to('cpu'), return_state=True
                    )

                    strongest_advex, _ = current_pgd_rr.attack(current_x, current_y, hx)

                    x_adv[idx, :, :] = strongest_advex.detach().numpy()
            if test_counter <= len(control_seqs) / 2:
                x0_seq.append(x0)
                y0_seq.append(y0)
                x_seq.append(x)
                y_seq.append(y)
                x_adv_seq.append(x)

            if test_counter > len(control_seqs)/2:
                x0_seq.append(x0)
                y0_seq.append(y0)
                x_seq.append(x)
                y_seq.append(y)
                x_adv_seq.append(x_adv)

        return np.vstack(x0_seq), np.vstack(y0_seq), np.vstack(x_adv_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'x0': self.x0[idx],
            'y0': self.y0[idx],
            'x': self.x[idx],
            'y': self.y[idx],
        }
