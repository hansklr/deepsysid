from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from deepsysid.networks.rnn import HybridLinearizationRnn, Linear, LureSystem

# torch.set_default_dtype(d=torch.float64)

n_batch = 3  # batch size
nu = 1  # input size
nx = 4  # state size
ny = 1
N = 5  # samples
nw = 10
nz = nw


def get_linear_matrices() -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    A = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-0.5, -1, -4, -0.2]]
    B = [[0.0], [0.0], [0.0], [1.0]]
    C = [[1.0, 0.0, 0.0, 0.0]]
    D = [[0.0]]
    return (np.array(A), np.array(B), np.array(C), np.array(D))


def get_lure_matrices() -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    A, B1, C1, D11 = get_linear_matrices()
    B2 = torch.rand(size=(nx, nw)).float()
    C2 = torch.rand(size=(nz, nx)).float()
    D12 = torch.rand(size=(ny, nw)).float()
    D21 = torch.rand(size=(nz, nu)).float()
    return (
        torch.tensor(A).float(),
        torch.tensor(B1).float(),
        B2,
        torch.tensor(C1).float(),
        torch.tensor(D11).float(),
        D12,
        C2,
        D21,
    )


def test_linear_dynamics() -> None:
    u = torch.zeros(size=(n_batch, N, nu, 1))
    u[0, 0, :, :] = torch.tensor(data=[[-4]]).float()
    x0 = torch.zeros(size=(n_batch, nx, 1)).float()
    x0[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]]).float()
    A, B, C, D = get_linear_matrices()

    model = Linear(
        A=torch.tensor(A).float(),
        B=torch.tensor(B).float(),
        C=torch.tensor(C).float(),
        D=torch.tensor(D).float(),
    ).float()
    x1 = model.state_dynamics(
        x=x0,
        u=u[:, 0, :, :],
    )

    y1 = model.output_dynamics(x=x1, u=u[:, 1, :, :])
    assert x1.shape == (n_batch, nx, 1)
    assert y1.shape == (n_batch, ny, 1)


def test_linear_forward() -> None:
    u = torch.zeros(size=(n_batch, N, nu, 1))
    u[0, 0, :, :] = torch.tensor(data=[[-4]])
    x0 = torch.zeros(size=(n_batch, nx, 1))
    x0[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
    A, B, C, D = get_linear_matrices()

    model = Linear(
        A=torch.tensor(A).float(),
        B=torch.tensor(B).float(),
        C=torch.tensor(C).float(),
        D=torch.tensor(D).float(),
    )
    _, x = model.forward(x0=x0, us=u, return_state=True)
    y = model.forward(x0=x0, us=u)
    assert y.shape == (n_batch, N, ny, 1)
    assert x.shape == (n_batch, N + 1, nx, 1)


def test_lure_forward() -> None:
    u = torch.zeros(size=(n_batch, N, nu, 1))
    u[0, 0, :, :] = torch.tensor(data=[[-4]])
    x0 = torch.zeros(size=(n_batch, nx, 1))
    x0[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
    alpha = 0
    beta = 1

    def Delta_tilde(z: torch.Tensor) -> torch.Tensor:
        return (2 / beta - alpha) * (torch.tanh(z) - ((alpha + beta) / 2) * z)

    model = LureSystem(
        *get_lure_matrices(), Delta=Delta_tilde, device=torch.device('cpu')
    )

    _, x, w = model.forward(x0=x0, us=u, return_states=True)
    y = model.forward(x0=x0, us=u)

    assert y.shape == (n_batch, N, ny, 1)
    assert x.shape == (n_batch, N + 1, nx, 1)
    assert w.shape == (n_batch, N, nw, 1)


def test_hybrid_linearization_rnn_init() -> None:
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )


def test_hybrid_linearization_rnn_forward() -> None:

    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.set_lure_system()
    u = torch.zeros(size=(n_batch, N, nu))
    u[0, :, :] = torch.tensor(np.float64(np.arange(N)).reshape(N, nu))
    x0_lin = torch.zeros(size=(n_batch, nx, 1))
    x0_lin[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
    x0_rnn = torch.zeros(size=(n_batch, nx, 1))

    y_hat, (x, w) = model.forward(x_pred=u, hx=(x0_lin, x0_rnn))
    assert y_hat.shape == (n_batch, N, ny)
    assert x.shape == (n_batch, N + 1, nx * 2)
    assert w.shape == (n_batch, N, nw)


def test_hybrid_linearization_rnn_backward() -> None:
    L = torch.nn.MSELoss()
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.set_lure_system()
    opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    y = torch.zeros(size=(n_batch, N, ny))
    u = torch.zeros(size=(n_batch, N, nu))
    u[0, :, :] = torch.tensor(np.float64(np.arange(N)).reshape(N, nu))
    x0_lin = torch.zeros(size=(n_batch, nx, 1))
    x0_lin[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
    x0_rnn = torch.zeros(size=(n_batch, nx, 1))

    y_hat, (x, w) = model.forward(x_pred=u, hx=(x0_lin, x0_rnn))
    loss = L(y, y_hat)
    loss.backward()
    opt.step()


# def test_hybrid_linearization_rnn_backward_barrier() -> None:
#     L = torch.nn.MSELoss()
#     A_lin, B_lin, C_lin, _ = get_linear_matrices()
#     model = HybridLinearizationRnn(
#         A_lin=A_lin,
#         B_lin=B_lin,
#         C_lin=C_lin,
#         alpha=0,
#         beta=1,
#         nwu=nw,
#         nzu=nz,
#         gamma=1.0,
#     )
#     model.set_lure_system()
#     opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)
#     # y = torch.zeros(size=(n_batch, N, ny))
#     # u = torch.zeros(size=(n_batch, N, nu))
#     # u[0, :, :] = torch.tensor(np.float64(np.arange(N)).reshape(N, nu))
#     # x0_lin = torch.zeros(size=(n_batch, nx, 1))
#     # x0_lin[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
#     # x0_rnn = torch.zeros(size=(n_batch, nx, 1))
#     barrier = model.get_barrier(1)
#     P = model.get_constraints().detach().numpy()
#     nxi = 2 * model.nx
#     nzp = model.nzp
#     nwp = model.nwp
#     nzu = model.nzu
#     nwu = model.nwu
#     P_lambda = np.zeros((2 * (nxi + nzp + nzu), 2 * (nxi + nzp + nzu)))
#     P_lambda[nxi + nzp : nxi + nzp + nzu, nxi + nwp : nxi + nwp + nwu] = -np.eye(nwu)
#     P_lambda[nxi + nzp + nzu + nxi + nzp :, nxi + nwp + nwu + nxi + nwp :] = -np.eye(
#         nwu
#     )
#     P_grad_lambda = -np.trace(P_lambda @ np.linalg.inv(P)) * np.eye(nwu)
#     barrier.backward()
#     print(
#         f'lam grad {model.lam.grad.detach().numpy()}, '
#         f'P_grad_lam {np.diag(P_grad_lambda)}'
#     )

#     assert (
#         np.linalg.norm(torch.diag(model.lam.grad).detach().numpy() - P_grad_lambda)
#         < 1e-4
#     )


def test_hybrid_linearization_rnn_backward_two_steps() -> None:
    torch.autograd.set_detect_anomaly(True)
    L = torch.nn.MSELoss()
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.set_lure_system()
    opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    dataset = []

    y = torch.zeros(size=(n_batch, N, ny))
    u = torch.zeros(size=(n_batch, N, nu))
    u[0, :, :] = torch.tensor(np.float64(np.arange(N)).reshape(N, nu))
    x0_lin = torch.zeros(size=(n_batch, nx, 1))
    x0_lin[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
    x0_rnn = torch.zeros(size=(n_batch, nx, 1))
    dataset.append({'y': y, 'u': u, 'x0_lin': x0_lin, 'x0_rnn': x0_rnn})

    y = torch.zeros(size=(n_batch, N, ny))
    u = torch.zeros(size=(n_batch, N, nu))
    u[0, :, :] = torch.tensor(np.float64(np.arange(5, 5 + N)).reshape(N, nu))
    x0_lin = torch.zeros(size=(n_batch, nx, 1))
    x0_lin[0, :, :] = torch.tensor(data=[[-4], [0.7], [-0.5], [4]])
    x0_rnn = torch.zeros(size=(n_batch, nx, 1))
    dataset.append({'y': y, 'u': u, 'x0_lin': x0_lin, 'x0_rnn': x0_rnn})
    model.train()

    for batch in dataset:
        model.zero_grad()
        y_hat, (x, w) = model.forward(
            x_pred=batch['u'], hx=(batch['x0_lin'], batch['x0_rnn'])
        )
        loss = L(batch['y'], y_hat)
        loss.backward()
        opt.step()
        model.set_lure_system()
    # assert False


def test_hybrid_linearization_rnn_parameters() -> None:
    Omega_tilde = [
        'L_x_flat',
        'L_y_flat',
        'lam',
        'K',
        'L1',
        'L2',
        'L3',
        'M1',
        'N11',
        'N12',
        'N13',
        'M2',
        'N21',
        'N22',
        'N23',
    ]

    L = torch.nn.MSELoss()
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.set_lure_system()
    y = torch.zeros(size=(n_batch, N, ny))
    u = torch.zeros(size=(n_batch, N, nu))
    u[0, :, :] = torch.tensor(np.double(np.arange(N)).reshape(N, nu))
    x0_lin = torch.zeros(size=(n_batch, nx, 1))
    x0_lin[0, :, :] = torch.tensor(data=[[-1], [1], [0.5], [-0.5]])
    x0_rnn = torch.zeros(size=(n_batch, nx, 1))

    y_hat, (x, w) = model.forward(x_pred=u, hx=(x0_lin, x0_rnn))
    loss = L(y, y_hat)

    loss.backward()
    for idx, p in enumerate(model.named_parameters()):
        assert Omega_tilde[idx] == p[0]
        assert p[1].grad is not None


def test_hybrid_linearization_rnn_constraints() -> None:

    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.set_lure_system()
    P = model.get_constraints()
    # check if P is symmetric
    P_test = P.detach().clone()
    assert torch.linalg.norm(P - 0.5 * (P_test.T + P_test)) < 1e-10
    assert P.shape == (38, 38)


def test_hybrid_linearization_rnn_check_constraints() -> None:
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.initialize_lmi()
    assert model.check_constraints()


def test_hybrid_linearization_rnn_construct_lower_triangular_matrix() -> None:
    L_test = torch.tensor(
        [[0, 0, 0, 0], [4, 1, 0, 0], [7, 5, 2, 0], [9, 8, 6, 3]]
    ).float()
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    diag_length = 4
    L_flat = torch.tensor(np.arange(10)).float()
    L = model.construct_lower_triangular_matrix(L_flat, diag_length)

    assert torch.linalg.norm(L_test - L) < 1e-10


def test_hybrid_linearization_rnn_extract_vector_from_lower_triangular_matrix() -> None:
    L_test = np.arange(10, dtype=np.float64)
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    L = np.array(
        [[0, 0, 0, 0], [4, 1, 0, 0], [7, 5, 2, 0], [9, 8, 6, 3]], dtype=np.float64
    )
    L_flat = model.extract_vector_from_lower_triangular_matrix(L)

    assert np.linalg.norm(L_test - L_flat) < 1e-10


def test_hybrid_linearization_rnn_projection() -> None:
    A_lin, B_lin, C_lin, _ = get_linear_matrices()
    model = HybridLinearizationRnn(
        A_lin=A_lin,
        B_lin=B_lin,
        C_lin=C_lin,
        alpha=0,
        beta=1,
        nwu=nw,
        nzu=nz,
        gamma=1.0,
    )
    model.K.data = model.K.data + torch.eye(model.nx) * 20.0
    assert not model.check_constraints()
    model.project_parameters()
    assert model.check_constraints()
