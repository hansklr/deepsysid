from typing import Tuple

import torch
import torch.nn as nn

from deepsysid.models.adversarial.adversaries.base_adversary import Adversary
from deepsysid.models.recurrent import LSTMInitModel

device = 'cpu'


class Fgsm(Adversary):
    def __init__(
            self,
            lstm: LSTMInitModel,
            step_size: float
    ):
        self.lstm = lstm
        self.step_size = step_size

    def attack(
            self,
            control: torch.Tensor,
            state: torch.Tensor,
            init_hx: Tuple[torch.Tensor],
    ):
        '''
        Performs FGSM with the given sequences.
        Expects normalized sequences as input.
        '''

        control = control.unsqueeze(0).float().detach().requires_grad_(True).to(device)
        state = state.unsqueeze(0)

        mse = nn.MSELoss().to(device)
        loss = mse(
            self.lstm.predictor.forward(control, hx=init_hx, return_state=False),
            state
        )
        loss.backward(retain_graph=True)

        grad = control.grad.squeeze()
        advex = control + (self.step_size * grad.sign())

        adversarial_state = self.lstm.predictor.forward(
            advex,
            hx=init_hx,
            return_state=False
        )

        achieved_loss = mse(adversarial_state, state).item()
        print(f'Achieved step loss: {achieved_loss}')

        return advex.squeeze(), achieved_loss


class Pgd(Adversary):
    def __init__(
            self,
            lstm: LSTMInitModel,
            steps: int,
            step_size: float,
            eps: float,
            perturbation_type: str = 'per_single_value'
    ):
        self.lstm = lstm
        self.steps = steps
        self.step_size = step_size
        self.eps = eps
        self.perturbation_type = perturbation_type

    def attack(
        self,
        control: torch.Tensor,
        state: torch.Tensor,
        init_hx: Tuple[torch.Tensor],
    ):
        '''
        Performs PGD with the given sequences.
        This is achieved by performing FGSM multiple times.
        If the adversarial example leaves the search space, defined by epsilon, during the updates,
        then it is projected back into it.
        Expects normalized sequences as input.
        '''

        fgsm = Fgsm(
            lstm=self.lstm,
            step_size=self.step_size
        )

        advex = control
        strongest_advex = advex
        biggest_loss = 0
        for step in range(self.steps):
            print(f'Step: {step+1}')

            advex, achieved_loss = fgsm.attack(
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

            control_tensor_update = control + perturbation

            if achieved_loss > biggest_loss:
                biggest_loss = achieved_loss
                strongest_advex = control_tensor_update

        return strongest_advex, biggest_loss


class PgdRR(Adversary):
    def __init__(
            self,
            lstm: LSTMInitModel,
            steps: int,
            step_size: float,
            eps: float,
            restarts: int = 1,
            perturbation_type: str = 'per_single_value',
            control_seq_size: int = 900
    ):
        self.lstm = lstm
        self.steps = steps
        self.step_size = step_size
        self.eps = eps
        self.restarts = restarts
        self.perturbation_type = perturbation_type
        self.control_seq_size = control_seq_size

    def attack(
            self,
            control: torch.Tensor,
            state: torch.Tensor,
            init_hx: Tuple[torch.Tensor],
    ):
        '''
        Performs PGD with random restarts with the given sequences.
        This is achieved by performing PGD multiple times on multiple randomly initialized perturbed control sequences.
        Expects normalized sequences as input.
        '''

        pgd = Pgd(
            lstm=self.lstm,
            steps=self.steps,
            step_size=self.step_size,
            eps=self.eps,
            perturbation_type=self.perturbation_type
        )

        advexs = [(control, 0)]
        for restart in range(self.restarts):
            print(f'Restart: {restart+1}')

            # Initialize a random starting point of the restart.
            random_perturbation = torch.FloatTensor(self.control_seq_size, 4).uniform_(-self.eps, self.eps)
            advex = control + random_perturbation

            # Create adversarial example with PGD.
            advex, achieved_loss = pgd.attack(
                advex,
                state,
                init_hx
            )

            # Add created advex to results
            advexs.append((advex, achieved_loss))

            # Sort strongest adversarial examples by loss value.
            # After sorting advexs[0] is the strongest adversary created.
            advexs = sorted(advexs, key=lambda tup: tup[1], reverse=True)

            print(advexs)

            print(f'Achieved loss of the restart: {achieved_loss}')
            print(f'Biggest loss of all restarts so far: {(advexs[0])[1]}')

        # Return the strongest created adversarial examples and their achieved losses.
        return advexs
