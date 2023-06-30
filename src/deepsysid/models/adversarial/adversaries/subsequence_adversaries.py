from typing import Tuple

import torch
import torch.nn as nn

from deepsysid.models.adversarial.adversaries.base_adversary import Adversary
from deepsysid.models.recurrent import LSTMInitModel

device = torch.device('cpu')

class SubseqFgsm(Adversary):
    def __init__(
            self,
            lstm: LSTMInitModel,
            step_size: float,
            subseq: Tuple[int, int]
    ):
        self.lstm = lstm
        self.step_size = step_size
        self.subseq_start = subseq[0]
        self.subseq_end = subseq[1]

    def attack(
            self,
            control: torch.Tensor,
            state: torch.Tensor,
            init_hx: Tuple[torch.Tensor],
            control_seq_size: int = 900
    ):
        '''
        Performs FGSM with the given sequences.
        Only a given subsequence is being perturbed.
        Expects normalized sequences as input.
        '''

        # Slicing the control sequence.
        control_subseq = control[self.subseq_start:self.subseq_end, :]
        control_left = control[0:self.subseq_start, :]
        control_right = control[self.subseq_end:control_seq_size, :]

        control_subseq = control_subseq.float().detach().requires_grad_(True).to(device)
        state = state.unsqueeze(0)

        # Calculation of MSE between true state and state prediction and backward pass.
        mse = nn.MSELoss().to(device)
        loss = mse(
            self.lstm.predictor.forward(
                torch.concat((
                    control_left,
                    control_subseq,
                    control_right
                )).unsqueeze(0),
                hx=init_hx,
                return_state=False
            ),
            state
        )
        loss.backward(retain_graph=True)

        # Perturbing the subsequence by taking a step in the direction of its gradient.
        grad = control_subseq.grad.squeeze()
        advex_subseq = control_subseq + (self.step_size * grad.sign())

        advex = torch.concat((
            control_left,
            advex_subseq,
            control_right
        )).unsqueeze(0)

        # Predicting state variables based on the adversarial example.
        adversarial_state = self.lstm.predictor.forward(
            advex,
            hx=init_hx,
            return_state=False
        )

        # Loss between true state and the prediction of the adversarial example.
        achieved_loss = mse(adversarial_state, state).item()

        print(f'Achieved step loss: {achieved_loss}')

        # Return adversarial example and the achieved MSE.
        return torch.concat((control_left,
                             advex_subseq,
                             control_right)), achieved_loss


class SubseqPgd(Adversary):
    def __init__(
            self,
            lstm: LSTMInitModel,
            steps: int,
            step_size: float,
            eps: float,
            subseq: Tuple[int, int],
            perturbation_type: str = 'per_single_value'
    ):
        self.lstm = lstm
        self.steps = steps
        self.step_size = step_size
        self.eps = eps
        self.subseq = subseq
        self.perturbation_type = perturbation_type

    def attack(
            self,
            control: torch.Tensor,
            state: torch.Tensor,
            init_hx: Tuple[torch.Tensor]
    ):
        '''
        Performs PGD with the given sequences.
        This is achieved by performing FGSM multiple times.
        If the adversarial example leaves the search space, defined by epsilon, during the updates,
        then it is projected back into it.
        Only a given subsequence is being perturbed.
        Expects normalized sequences as input.
        '''

        subseq_fgsm = SubseqFgsm(
            lstm=self.lstm,
            step_size=self.step_size,
            subseq=self.subseq
        )
        achieved_loss = 0
        advex = control
        for step in range(self.steps):
            print(f'Step: {step+1}')
            advex, achieved_loss = subseq_fgsm.attack(
                advex,
                state,
                init_hx
            )

            perturbation = advex - control

            # Projection depending on projection function.
            # Allow a certain amount of perturbation for each value of the sequence.
            # Note that the values are normalized, so a single real valued epsilon is sufficient.
            if self.perturbation_type == 'per_single_value':
                perturbation = torch.clamp(perturbation, -self.eps, self.eps)

            # Allow a certain amount of perturbation for each sequence component (control variable) of the sequence.
            elif self.perturbation_type == 'per_seq_component':
                for component in range(self.lstm.control_dim):
                    perturbation_component = perturbation[:, component]
                    if torch.sum(torch.abs(perturbation_component)) > torch.numel(perturbation_component) * self.eps:
                        perturbation_component -= perturbation_component.sign() * self.step_size * 2
                        perturbation[:, component] = perturbation_component

            # Allow a certain amount of perturbation for the sum of all values of the sequence.
            elif self.perturbation_type == 'total':
                if torch.sum(torch.abs(perturbation)) > torch.numel(perturbation) * self.eps:
                    perturbation -= perturbation.sign() * self.step_size * 2

            else:
                raise ValueError('perturbation type can only be "per_single_value", "per_seq_component" or "total"')

            advex = control + perturbation


        # Return adversarial example and the achieved MSE.
        return advex, achieved_loss


class SubseqPgdRR(Adversary):
    def __init__(
            self,
            lstm: LSTMInitModel,
            subseq: Tuple[int, int],
            step_size: float,
            eps: float,
            steps: int = 1,
            restarts: int = 1,
            perturbation_type: str = 'per_single_value'
    ):
        self.lstm = lstm
        self.subseq = subseq
        self.subseq_start = subseq[0]
        self.subseq_end = subseq[1]
        self.step_size = step_size
        self.eps = eps
        self.steps = steps
        self.restarts = restarts
        self.perturbation_type = perturbation_type

    def attack(
            self,
            control: torch.Tensor,
            state: torch.Tensor,
            init_hx: Tuple[torch.Tensor],
            sequence_length: int = 900,
            num_of_variables: int = 4
    ):
        '''
        Performs PGD with random restarts with the given sequences.
        This is achieved by performing PGD multiple times on multiple randomly initialized perturbed control sequences.
        Only a given subsequence is being perturbed.
        Expects normalized sequences as input.
        '''

        subseq_pgd = SubseqPgd(
            lstm=self.lstm,
            steps=self.steps,
            step_size=self.step_size,
            eps=self.eps,
            subseq=self.subseq,
            perturbation_type=self.perturbation_type
        )

        advexs = [(control, 0)]
        for restart in range(self.restarts):
            print(f'Restart: {restart+1}')

            # Initialize a random starting point of the restart.
            subseq_length = self.subseq_end - self.subseq_start + 1
            random_subseq_perturbation = torch.FloatTensor(subseq_length, 4).uniform_(-self.eps, self.eps)
            random_perturbation = torch.concat((
                torch.zeros(self.subseq_start, 4),
                random_subseq_perturbation,
                torch.zeros(sequence_length - self.subseq_end - 1, 4)
            ))
            advex = control + random_perturbation
            # Create adversarial example with PGD.
            advex, achieved_loss = subseq_pgd.attack(
                advex,
                state,
                init_hx
            )

            # Add created advex to results
            advexs.append((advex, achieved_loss))

            # Sort strongest adversarial examples by loss value.
            # After sorting advexs[0] is the strongest adversary created.
            advexs = sorted(advexs, key=lambda tup: tup[1], reverse=True)

            print(f'Achieved loss of the restart: {achieved_loss}')
            print(f'Biggest loss of all restarts so far: {(advexs[0])[1]}')

        # Return the strongest created adversarial examples and their achieved losses.
        return advexs
