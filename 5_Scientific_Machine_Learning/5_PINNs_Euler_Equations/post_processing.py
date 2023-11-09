# Import packages
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import forward_PINN
from analytic_solutions import SodShockAnalytic

# Import outcomes from training

results_folder = 'results/'
subfolders = [f.path for f in os.scandir(results_folder) if f.is_dir()]

models_dict = {}

for i, subfolder in enumerate(subfolders):
    try:
        # Load model data
        with open(os.path.join(subfolder, 'data_dict.pkl'), 'rb') as f:
            model_data_dict = pickle.load(f)

        # Load history training
        with open(os.path.join(subfolder, 'loss_history.pkl'), 'rb') as f:
            loss_history = pickle.load(f)

        model = forward_PINN.ffnn(nn_width=model_data_dict['width'],
                                  nn_hidden=model_data_dict['hidden'],
                                  activation_type=model_data_dict['activation'])
        model.load_state_dict(torch.load(os.path.join(subfolder, 'model.pth')))
        model.eval()

        models_dict[i] = {'model': model,
                          'model_data_dict': model_data_dict,
                          'loss_history': loss_history}
    except FileNotFoundError:
        print('FileNotFoundError')
        continue


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
    for k in models_dict.keys():
        model = models_dict[k]['model']
        U_pred = model(tx_test)
        rho_pred = U_pred[:, 0].detach().cpu().numpy()
        u_pred = U_pred[:, 1].detach().cpu().numpy()
        p_pred = U_pred[:, 2].detach().cpu().numpy()

        models_dict[k]['rho_pred'] = rho_pred
        models_dict[k]['u_pred'] = u_pred
        models_dict[k]['p_pred'] = p_pred


#%%
fig, ax = plt.subplots(2,2 , figsize=(10, 8))
for k in models_dict.keys():
    ax[0, 0].plot(models_dict[k]['loss_history'], label=models_dict[k]['model_data_dict']['case_name'])
ax[0, 0].set_xlabel('Epochs')
ax[0, 0].set_yscale('log')
ax[0, 0].set_ylabel('Loss')
ax[0, 0].legend()

ax[0, 1].plot(x_exact, rho_exact, 'k--', label='Analytic')
for k in models_dict.keys():
    ax[0, 1].plot(x_test.numpy(), models_dict[k]['rho_pred'], label=models_dict[k]['model_data_dict']['case_name'])
ax[0, 1].set_xlabel(r'$x$')
ax[0, 1].set_ylabel(r'$\rho$')
ax[0, 1].legend()

ax[1, 0].plot(x_exact, u_exact, 'k--', label='Analytic')
for k in models_dict.keys():
    ax[1, 0].plot(x_test.numpy(), models_dict[k]['u_pred'], label=models_dict[k]['model_data_dict']['case_name'])
ax[1, 0].set_xlabel(r'$x$')
ax[1, 0].set_ylabel(r'$u$')
ax[1, 0].legend()

ax[1, 1].plot(x_exact, p_exact, 'k--', label='Analytic')
for k in models_dict.keys():
    ax[1, 1].plot(x_test.numpy(), models_dict[k]['p_pred'], label=models_dict[k]['model_data_dict']['case_name'])
ax[1, 1].set_xlabel(r'$x$')
ax[1, 1].set_ylabel(r'$p$')
ax[1, 1].legend()


plt.tight_layout()
plt.show()


