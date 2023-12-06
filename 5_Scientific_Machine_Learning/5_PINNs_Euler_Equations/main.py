__author__ = 'Dario Rodriguez'

import numpy as np
import torch
from numerical_solution import solve_Euler_PDE_WENO
from analytic_solutions import SodShockAnalytic
from aux_functions import (load_single_trained_model, load_all_trained_models,
                           create_animation, generate_comparative_plot, eval_nn_model,
                           generate_colormesh_plot, compute_error_at_timesteps,
                           generate_error_plot)


# ---------------------------------------------
#                   Inputs
# ---------------------------------------------
nx = 100    # Number of points in the space discretization
T = 0.2     # Final time to compute all solutions
x_vec = np.linspace(0, 1, nx)


# -----------------------------------------------
#               Numerical solution
# -----------------------------------------------
# Note: this was computed first because t_vec is dynamically computed within
# this function and is needed for the analytic and neural network solutions
# for a proper comparison

x_weno, U_final_weno, data_weno = solve_Euler_PDE_WENO(nx, T)
t_vec = data_weno['t']
rho_num, u_num, p_num = data_weno['rho'], data_weno['u'], data_weno['p']
U_num = np.array([rho_num, u_num, p_num])


# -----------------------------------------------
#              Analytic solution
# -----------------------------------------------
x_exact = x_vec.copy()
rho_exact, u_exact, p_exact = [], [], []
for t in t_vec:
    u_exact_i = SodShockAnalytic(x_exact, t)
    rho_exact.append(u_exact_i[0, :])
    u_exact.append(u_exact_i[1, :])
    p_exact.append(u_exact_i[2, :])
U_exact = np.array([rho_exact, u_exact, p_exact])
U_exact_final = U_exact[:, -1, :]


# -----------------------------------------------
#             Neural network solution
# -----------------------------------------------
# General settings:
device = torch.device('cpu')

# -----------------------------------------------
#                   CASE 1
# No extension domain, no weights, original NN architecture
# -----------------------------------------------
model_path_C1 = ('neural_network_solutions/training_results/'
                     'case_ext_0_w_0_nn__w_30_h_7_act_tanh_lr_0.0005_epo_80000_samp_uniform_t_1000_x_1000_per_11_bc_1000')

model_dict_C1 = load_single_trained_model(model_path_C1, device=device)
nn_model_C1, metadata_C1, loss_C1 = (model_dict_C1[0]['model'], model_dict_C1[0]['model_metadata'],
                                     model_dict_C1[0]['loss_history'])

# Comparative plot at T = 0.2
T = 0.2
x_nn = torch.linspace(0, 1, nx)[:, None].to(device)
model_dict_C1 = eval_nn_model(model_dict_C1, x_nn, T, device='cpu')
fig_c1_0, ax_c1_0 = generate_comparative_plot(x_nn, model_dict_C1, x_exact, U_exact_final, x_weno, U_final_weno,
                                    custom_title=None, custom_legend=[''], legend_loss=True)

rho_nn, u_nn, p_nn = [], [], []

for t in t_vec:
    t_i = t * torch.ones_like(x_nn).to(device)
    tx_i = torch.cat((t_i, x_nn), 1).to(device)
    U_nn_i = nn_model_C1(tx_i)

    rho_nn.append(U_nn_i[:, 0].detach().cpu().numpy())
    u_nn.append(U_nn_i[:, 1].detach().cpu().numpy())
    p_nn.append(U_nn_i[:, 2].detach().cpu().numpy())
U_nn_C1 = np.array([rho_nn, u_nn, p_nn])

# Generate colormesh plot
fig_c1_1, ax_c1_1 = generate_colormesh_plot(SodShockAnalytic, nn_model_C1)

# Generate MSE error plot
error_c1 = compute_error_at_timesteps(U_exact, U_nn_C1, U_num)
error_ls_c1 = [error_c1]
fig_c1_2, ax_c1_2 = generate_error_plot(t_vec, error_ls_c1, ['Case 1'])

# Generate animation
anim_c1 = create_animation(x_vec, U_exact, U_num, U_nn_C1)


# -----------------------------------------------
#                 LOAD ALL MODELS
# -----------------------------------------------
T = 0.2
trained_models_dict = load_all_trained_models(device='cpu')
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
trained_models_dict = eval_nn_model(trained_models_dict, x_test, T, device='cpu')
# -----------------------------------------------

# -----------------------------------------------
#                   CASE 2
# Domain extension, loss weights, original NN architecture
# -----------------------------------------------

# Filter trained models by extension domain and weights with original NN architecture
case_2_dict = {}
for k, v in trained_models_dict.items():
    if (v['model_metadata']['width'] == 30 and v['model_metadata']['hidden'] == 7 and
        v['model_metadata']['activation'] == 'tanh' and v['model_metadata']['lr'] == 0.0005 and
        v['model_metadata']['n_epochs'] == 50000 and v['model_metadata']['sampling_mode'] == 'uniform' and
        v['model_metadata']['percent_int_points'] == 15):

        case_2_dict[k] = v

legend_list_c2 = ['Ext. Dom', 'Ext. Dom + W', 'W', 'Original']
fig_c2_0, ax_c2_0 = generate_comparative_plot(x_test, case_2_dict, x_exact, U_exact_final, x_weno, U_final_weno,
                                    custom_title=None, custom_legend=legend_list_c2)

nn_model_C2 = case_2_dict[17]['model']  # Ext. Dom + W
rho_nn, u_nn, p_nn = [], [], []

for t in t_vec:
    t_i = t * torch.ones_like(x_nn).to(device)
    tx_i = torch.cat((t_i, x_nn), 1).to(device)
    U_nn_i = nn_model_C2(tx_i)

    rho_nn.append(U_nn_i[:, 0].detach().cpu().numpy())
    u_nn.append(U_nn_i[:, 1].detach().cpu().numpy())
    p_nn.append(U_nn_i[:, 2].detach().cpu().numpy())
U_nn_C2 = np.array([rho_nn, u_nn, p_nn])

# Generate colormesh plot
fig_c2_1, ax_c2_1 = generate_colormesh_plot(SodShockAnalytic, nn_model_C2)

# Generate MSE error plot
error_c2 = compute_error_at_timesteps(U_exact, U_nn_C2, U_num)
error_ls_c2 = [error_c1, error_c2]
fig_c1_2, ax_c1_2 = generate_error_plot(t_vec, error_ls_c2)

# Generate animation
anim_c2 = create_animation(x_vec, U_exact, U_num, U_nn_C2)

# -----------------------------------------------
#                   CASE 3
# - Clustered sampling
# - Domain extension, loss weights, original NN architecture
# -----------------------------------------------
model_path_C3 = ('neural_network_solutions/training_results/'
                 'case_ext_1_w_1_nn__w_30_h_7_act_tanh_lr_0.0005_epo_90000_samp_clustered_t_1000_x_1000_per_20_bc_1000')

model_dict_C3 = load_single_trained_model(model_path_C3, device=device)
model_dict_C3 = eval_nn_model(model_dict_C3, x_nn, T, device=device)

case_3_dict = case_2_dict.copy()
case_3_dict[0] = model_dict_C3[0]

legend_list_c3 = ['Ext. Dom', 'Ext. Dom + W', 'W', 'Original', 'Clustered']
fig_c3_0, ax_c3_0 = generate_comparative_plot(x_test, case_3_dict, x_exact, U_exact_final, x_weno, U_final_weno,
                                    custom_title=None, custom_legend=legend_list_c3)

nn_model_C3 = case_3_dict[0]['model']  # Clustered + Ext. Dom + W
rho_nn, u_nn, p_nn = [], [], []

for t in t_vec:
    t_i = t * torch.ones_like(x_nn).to(device)
    tx_i = torch.cat((t_i, x_nn), 1).to(device)
    U_nn_i = nn_model_C3(tx_i)

    rho_nn.append(U_nn_i[:, 0].detach().cpu().numpy())
    u_nn.append(U_nn_i[:, 1].detach().cpu().numpy())
    p_nn.append(U_nn_i[:, 2].detach().cpu().numpy())
U_nn_C3 = np.array([rho_nn, u_nn, p_nn])

# Generate colormesh plot
fig_c3_1, ax_c3_1 = generate_colormesh_plot(SodShockAnalytic, nn_model_C3)

# Generate MSE error plot
error_c3 = compute_error_at_timesteps(U_exact, U_nn_C3, U_num)
error_ls_c3 = [error_c1, error_c2, error_c3]
fig_c3_2, ax_c3_2 = generate_error_plot(t_vec, error_ls_c3)

# Generate animation
anim_c3 = create_animation(x_vec, U_exact, U_num, U_nn_C3)


# -----------------------------------------------
#                   CASE 4
# Generate a comparative plot for all cases
# -----------------------------------------------
fig_c4_0, ax_c4_0 = generate_comparative_plot(x_test, trained_models_dict,
                                              x_exact, U_exact_final,
                                              x_weno, U_final_weno,
                                              custom_title=None, custom_legend=None)

