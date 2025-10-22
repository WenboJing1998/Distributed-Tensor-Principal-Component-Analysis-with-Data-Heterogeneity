
## Simulation for the homogeneous setting, run on NYU High Performance Computing

import numpy as np
import pip
pip.main(["install", "tensorly"])
import tensorly as tl
import os

# singular_min: compute the minimum singular value over all matricizations of a tensor
# Input:
#       T: input tensor
# Output:
#       the minimum singular value over all matricizations of the input tensor

def singular_min(T):
    J = len(T.shape)
    sig_min = np.zeros(J)
    for j in range(J):
        T_j = tl.unfold(T, j)
        sig_min[j] = np.min(np.linalg.svd(T_j, full_matrices=False)[1])

    return(np.min(sig_min))

# rho: compute the distance between the column space of two matrices U and V
# Input:
#       U, V: two matrices of the same size
# Output:
#       the Frobenius norm of UU^T - VV^T

def rho(U, V):
    return(np.linalg.norm(U @ np.transpose(U) - V @ np.transpose(V), ord = "fro"))

# dist_tensor_homo: distributed tensor PCA for homogeneous tensors
# Input: 
#        T_list:    (list) tensors to be analysed.
#        r:         (integer) the prespecified rank for the principal component U of each mode.
#        U_inits:   (list) initial estimators for the principal components.
#        j_list:    (list) the indices of mode of interest, that is, only U_j with j in the j_list
#                   will be estimated.
# Output:
#       U_estmators:(list) estimators for the principal components.

def dist_tensor_homo(T_list, r, U_inits, j_list=[0]):
    
    J = len(T_list[0].shape)
    modes_all = list(range(J))
    L = len(T_list)
    U_hats = []
    for l in range(L):
        U_hat_l = []
        for j in range(J):
            if(j!=(J-1)):
                M = tl.unfold(tl.tenalg.multi_mode_dot(T_list[l], 
                                        [np.transpose(U) for U in (U_inits[l][:j] + U_inits[l][(j+1):])], 
                                        modes=modes_all[:j]+modes_all[(j+1): ]), j)
            else:
                M = tl.unfold(tl.tenalg.multi_mode_dot(T_list[l], 
                                        [np.transpose(U) for U in U_inits[l][:j]], 
                                        modes=modes_all[:j]), j)
                
            U_hat_jl = np.linalg.svd(M, full_matrices=False)[0][:, :r]
            U_hat_l.append(U_hat_jl)

        U_hats.append(U_hat_l)

    U_estmators = []
    for j in j_list:
        U_aggregate = 0
        for l in range(L):
            U_aggregate += U_hats[l][j] @ np.transpose(U_hats[l][j]) / L
        U_est = np.linalg.svd(U_aggregate , full_matrices=False)[0][:, :r]
        U_estmators.append(U_est)

    return(U_estmators)


# simulation: run the simulation experiments for the homogeneous tensors in the paper. Two different 
# methods are implemented for comparison: "distributed" (our proposed method) and "pooled." 
# Descriptions for the two methods are provided in the paper.
# Input: 
#        J:                 (integer) the number of the modes.
#        p:                 (integer) the dimension of each mode.
#        r:                 (integer) the prespecified rank for the common component U of each mode.
#        L:                 (integer) the number of tensors.
#        sigma:             (float) the noise level.
#        reps:              (integer) the number of repetitions. Defaults to be 100.
#        seed:              (integer) the random seed. Defaults to be 0.
# Output:
#       A tuple composed of four elements:
#       res_dist_all:       (numpy array) a vector of size reps recording the estimation error of the 
#                            "distributed" method in each repeat.
#       res_pooled_all:     (numpy array) a vector of size reps recording the estimation error of the 
#                           "pooled" method in each repeat.
#       res_inf_dist_all:   (numpy array) a vector of size reps recording the t-statistic of the 
#                            "distributed" method for inference in each repeat.
#       res_inf_pooled_all: (numpy array) a vector of size reps recording the t-statistic of the 
#                           "pooled" method for inference in each repeat.

def simulation(J, p, r, L, sigma, lambda_seq, reps=100, seed=0):

    np.random.seed(seed)

    modes_all = list(range(J))

    res_dist_all = []
    res_pooled_all = []
    res_inf_dist_all = []
    res_inf_pooled_all = []

    for lambda_min in lambda_seq:

        res_dist = []
        res_pooled = []
        res_inf_dist = []
        res_inf_pooled = []

        for rep in range(reps):
    
            # Generate U, G, T_star
            U_list = [np.linalg.qr(np.random.normal(size=(p, r)))[0] for _ in range(J)]
            G = np.random.normal(size=tuple([r] * J))
            G = G * lambda_min / singular_min(G) 
            T_star = tl.tucker_tensor.tucker_to_tensor((G, U_list))

            # Generate individual tensors
            Z_list = [np.random.normal(scale=sigma, size=tuple([p] * J)) for _ in range(L)]
            T_list = [T_star + Z for Z in Z_list]

            # Generate initial estimators
            U_inits = [tl.decomposition.tucker(T, rank=tuple([r] * J))[1] for T in T_list] 

            # Distributed Tensor PCA
            U_est = dist_tensor_homo(T_list, r, U_inits, j_list=modes_all)
            res_dist.append(rho(U_est[0], U_list[0]))

            # Inference
            UUt_est = [U @ np.transpose(U) for U in U_est]
            sigma_hat = np.linalg.norm((T_list[0] - tl.tenalg.multi_mode_dot(T_list[0], UUt_est, modes=modes_all)).reshape(-1)) / np.sqrt(p ** 3)
            M_1 = tl.unfold(tl.tenalg.multi_mode_dot(T_list[0], 
                                        [np.transpose(U) for U in (U_est[1:])], 
                                        modes=modes_all[1: ]), 0)

            Lambda_hat_1_inv = np.linalg.inv(np.diag(np.linalg.svd(M_1, full_matrices=False)[1][:r]))
            t_stat = (rho(U_est[0], U_list[0]) ** 2 - 2 * p * (sigma_hat ** 2) * (np.linalg.norm(Lambda_hat_1_inv, ord="fro") ** 2) / L) \
                        / (np.sqrt(8 * p) * (sigma_hat ** 2) * np.linalg.norm(Lambda_hat_1_inv * Lambda_hat_1_inv , ord="fro") / L)
            res_inf_dist.append(t_stat)


            # Pooled tensors for comparison
            T_pooled = 0
            for T in T_list:
                T_pooled += (1/L) * T
            U_init_pooled = [tl.decomposition.tucker(T_pooled, rank=tuple([r] * J))[1]]
            U_est_pooled = dist_tensor_homo([T_pooled], r, U_init_pooled, j_list=modes_all)
            res_pooled.append(rho(U_est_pooled[0], U_list[0]))

            UUt_est_pooled = [U @ np.transpose(U) for U in U_est_pooled]
            sigma_hat_pooled = np.linalg.norm((T_pooled - tl.tenalg.multi_mode_dot(T_pooled, UUt_est_pooled, modes=modes_all)).reshape(-1)) / np.sqrt(p ** 3)
            M_1_pooled = tl.unfold(tl.tenalg.multi_mode_dot(T_pooled, 
                                        [np.transpose(U) for U in (U_est_pooled[1:])], 
                                        modes=modes_all[1: ]), 0)

            Lambda_hat_1_inv_pooled = np.linalg.inv(np.diag(np.linalg.svd(M_1_pooled, full_matrices=False)[1][:r]))
            t_stat_pooled = (rho(U_est_pooled[0], U_list[0]) ** 2 - 2 * p * (sigma_hat_pooled ** 2) * (np.linalg.norm(Lambda_hat_1_inv_pooled, ord="fro") ** 2) ) \
                        / (np.sqrt(8 * p) * (sigma_hat_pooled ** 2) * np.linalg.norm(Lambda_hat_1_inv_pooled * Lambda_hat_1_inv_pooled , ord="fro") )
            res_inf_pooled.append(t_stat_pooled)

        # Record the results
        res_dist_all.append(res_dist)
        res_pooled_all.append(res_pooled)
        res_inf_dist_all.append(res_inf_dist)
        res_inf_pooled_all.append(res_inf_pooled)
    
    return(np.array(res_dist_all), np.array(res_pooled_all), np.array(res_inf_dist_all), np.array(res_inf_pooled_all))

L_seq = [10, 20]
p_seq = [50, 100, 200]
gamma_seq = np.linspace(0.45, 0.95, 20)

taskId = os.environ.get("SLURM_ARRAY_TASK_ID")
taskId = int(taskId) - 1

p_id = taskId % len(p_seq)
taskId1 = int((taskId - p_id) / len(p_seq))
L_id = taskId1 % len(L_seq)
gamma_id = int((taskId1 - L_id) / len(L_seq))

p = p_seq[p_id]
L = L_seq[L_id]
gamma = gamma_seq[gamma_id]


lambda_seq = [p ** gamma]
error_dist, error_pooled, inf_dist, inf_pooled = simulation(J=3, p=p, r=3, L=L, sigma=1, lambda_seq=lambda_seq, reps=1000, seed=0)

# Stack the output into a matrix of size reps * 4
res_mat = np.hstack((error_dist.reshape(-1, 1), error_pooled.reshape(-1, 1), inf_dist.reshape(-1, 1), inf_pooled.reshape(-1, 1)))

# Save the results
np.savetxt(f"results/res_p={p}_L={L}_gamma={gamma}.txt", res_mat)