# Import packages
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import forward_PINN
from analytic_solutions import SodShockAnalytic

# Import loss history
loss_folder = 'loss_history/'
files = ['loss_history.pkl', 'loss_history_w.pkl', 'loss_history_ext.pkl', 'loss_history_w_ext.pkl']
for file in files:
    with open(os.path.join(loss_folder, file), 'rb') as f:
        loaded_file = pickle.load(f)
        variable_name = file.split('.')[0]
        exec(variable_name + ' = loaded_file')

# Load models
model = forward_PINN.ffnn(30, 7)
model.load_state_dict(torch.load('trained_models/model.pth'))
model.eval()

model_w = forward_PINN.ffnn(30, 7)
model_w.load_state_dict(torch.load('trained_models/model_w.pth'))
model_w.eval()

model_ext = forward_PINN.ffnn(30, 7)
model_ext.load_state_dict(torch.load('trained_models/model_ext.pth'))
model_ext.eval()

model_w_ext = forward_PINN.ffnn(30, 7)
model_w_ext.load_state_dict(torch.load('trained_models/model_w_ext.pth'))
model_w_ext.eval()

# Compute analytic solutions
x_exact = np.linspace(0, 1, 100)
t_exact = 0.2
left_bc = np.array([1., 0., 1.])
right_bc = np.array([0.125, 0., 0.1])

U_exact = SodShockAnalytic(left_bc[0], left_bc[1], left_bc[2], right_bc[0], right_bc[1], right_bc[2], x_exact, 50, t_exact, 1.4)
rho_exact, u_exact, p_exact = U_exact[0, :], U_exact[1, :], U_exact[2, :]

# Set device
device = torch.device('cpu')
x_test = torch.linspace(0, 1, 100)[:, None].to(device)
t_test = 0.2 * torch.ones_like(x_test).to(device)
tx_test = np.hstack((t_test, x_test))
tx_test = torch.tensor(tx_test, dtype=torch.float32).to(device)

with torch.no_grad():
    # Compute predictions for plain model
    U_pred = model(tx_test)
    rho_pred = U_pred[:, 0].detach().cpu().numpy()
    u_pred = U_pred[:, 1].detach().cpu().numpy()
    p_pred = U_pred[:, 2].detach().cpu().numpy()

    # Compute prediction for weighted model
    U_pred_w = model_w(tx_test)
    rho_pred_w = U_pred_w[:, 0].detach().cpu().numpy()
    u_pred_w = U_pred_w[:, 1].detach().cpu().numpy()
    p_pred_w = U_pred_w[:, 2].detach().cpu().numpy()

    # Compute prediction for extended model
    U_pred_ext = model_ext(tx_test)
    rho_pred_ext = U_pred_ext[:, 0].detach().cpu().numpy()
    u_pred_ext = U_pred_ext[:, 1].detach().cpu().numpy()
    p_pred_ext = U_pred_ext[:, 2].detach().cpu().numpy()

    # Compute prediction for weighted and extended model
    U_pred_w_ext = model_w_ext(tx_test)
    rho_pred_w_ext = U_pred_w_ext[:, 0].detach().cpu().numpy()
    u_pred_w_ext = U_pred_w_ext[:, 1].detach().cpu().numpy()
    p_pred_w_ext = U_pred_w_ext[:, 2].detach().cpu().numpy()

#%%
fig, ax = plt.subplots(2,2 , figsize=(10, 8))
ax[0, 0].plot(loss_history, label='PINN')
ax[0, 0].plot(loss_history_w, label='PINN + W')
ax[0, 0].plot(loss_history_ext, label='PINN + Ext')
ax[0, 0].plot(loss_history_w_ext, label='PINN + W + Ext')
ax[0, 0].set_xlabel('Epochs')
ax[0, 0].set_yscale('log')
ax[0, 0].set_ylabel('Loss')
ax[0, 0].legend()

ax[0, 1].plot(x_test.numpy(), rho_pred, label='PINN')
ax[0, 1].plot(x_test.numpy(), rho_pred_w, label='PINN + W')
ax[0, 1].plot(x_test.numpy(), rho_pred_ext, label='PINN + Ext')
ax[0, 1].plot(x_test.numpy(), rho_pred_w_ext, label='PINN + W + Ext')
ax[0, 1].plot(x_exact, rho_exact, 'k--', label='Analytic')
ax[0, 1].set_xlabel(r'$x$')
ax[0, 1].set_ylabel(r'$\rho$')
ax[0, 1].legend()

ax[1, 0].plot(x_test.numpy(), u_pred, label='PINN')
ax[1, 0].plot(x_test.numpy(), u_pred_w, label='PINN + W')
ax[1, 0].plot(x_test.numpy(), u_pred_ext, label='PINN + Ext')
ax[1, 0].plot(x_test.numpy(), u_pred_w_ext, label='PINN + W + Ext')
ax[1, 0].plot(x_exact, u_exact, 'k--', label='Analytic')
ax[1, 0].set_xlabel(r'$x$')
ax[1, 0].set_ylabel(r'$u$')
ax[1, 0].legend()

ax[1, 1].plot(x_test.numpy(), p_pred, label='PINN')
ax[1, 1].plot(x_test.numpy(), p_pred_w, label='PINN + W')
ax[1, 1].plot(x_test.numpy(), p_pred_ext, label='PINN + Ext')
ax[1, 1].plot(x_test.numpy(), p_pred_w_ext, label='PINN + W + Ext')
ax[1, 1].plot(x_exact, p_exact, 'k--', label='Analytic')
ax[1, 1].set_xlabel(r'$x$')
ax[1, 1].set_ylabel(r'$p$')
ax[1, 1].legend()


plt.tight_layout()
plt.show()


