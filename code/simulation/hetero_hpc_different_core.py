
## Simulation for the heterogenous setting with different cores, run on NYU High Performance Computing

import numpy as np
import pip
pip.main(["install", "tensorly"])
import tensorly as tl
import matplotlib.pyplot as plt
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

# singular_max: compute the maximum singular value over all matricizations of a tensor
# Input:
#       T: input tensor
# Output:
#       the maximum singular value over all matricizations of the input tensor

def singular_max(T):
    
    J = len(T.shape)
    sig_max = np.zeros(J)
    for j in range(J):
        T_j = tl.unfold(T, j)
        sig_max[j] = np.max(np.linalg.svd(T_j, full_matrices=False)[1])

    return(np.max(sig_max))

# rho: compute the distance between the column space of two matrices U and V
# Input:
#       U, V: two matrices of the same size
# Output:
#       the Frobenius norm of UU^T - VV^T

def rho(U, V):
    return(np.linalg.norm(U @ np.transpose(U) - V @ np.transpose(V), ord = "fro"))


# dist_tensor_hetero: distributed tensor PCA for heterogeneous tensors
# Input: 
#        T_list:        (list) input tensors.
#        r_U:           (integer) the prespecified rank for the common component U of each mode.
#        r_V:           (integer) the prespecified rank for the individual component V of each mode.
#        UV_inits:      (list) initial estimators for the principal components [U V] of each mode.
#        j_list:        (list) the indices of mode of interest, that is, only U_j and V_j with j in 
#                       the j_list will be estimated.
#        l_list:        (list) the indices of tensors whose individual component V_j will be estimated.
# Output:
#        U_estmators:   (list) estimators for the common components.
#        V_estmators:   (list) estimators for the individual components.

def dist_tensor_hetero(T_list, r_U, r_V, UV_inits, j_list=[0], l_list=[0]):

    r = r_U + r_V
    J = len(T_list[0].shape)
    modes_all = list(range(J))
    L = len(T_list)
    U_hats = []
    M_list = []
    for l in range(L):
        # print(f"Begin Tensor {l}")
        U_hat_l = []
        M_list_l = []
        for j in range(J):
            # print(f"mode {j}")

            if(j!=(J-1)):
                M = tl.unfold(tl.tenalg.multi_mode_dot(T_list[l], 
                                        [np.transpose(UV) for UV in (UV_inits[l][:j] + UV_inits[l][(j+1):])], 
                                        modes=modes_all[:j]+modes_all[(j+1): ]), j)
            else:
                M = tl.unfold(tl.tenalg.multi_mode_dot(T_list[l], 
                                        [np.transpose(UV) for UV in UV_inits[l][:j]], 
                                        modes=modes_all[:j]), j)
                
            M_list_l.append(M)
            U_hat_jl = np.linalg.svd(M, full_matrices=False)[0][:, :r_U]
            U_hat_l.append(U_hat_jl)

        U_hats.append(U_hat_l)
        M_list.append(M_list_l)

    U_estmators = []
    V_estmators = []
    for j in j_list:
        U_aggregate = 0
        for l in range(L):
            U_aggregate += U_hats[l][j] @ np.transpose(U_hats[l][j]) / L
        U_est = np.linalg.svd(U_aggregate , full_matrices=False)[0][:, :r_U]
        U_estmators.append(U_est)

    for l in l_list:
        V_est_l = []
        for j in j_list:
            p_j = T_list[l].shape[j]
            V_hat_jl = np.linalg.svd((np.diag(np.ones(p_j)) - U_estmators[j] @ np.transpose(U_estmators[j])) @ M_list[l][j], full_matrices=False)[0][:, :r_V]
            V_est_l.append(V_hat_jl)
        V_estmators.append(V_est_l)


    return(U_estmators, V_estmators)


# simulation: run the simulation experiments for the heterogeneous tensors with the different cores. Three 
# different methods are implemented for comparison: "distributed" (our proposed method), "single," 
# and "pooled." Descriptions for the three methods are provided in the paper. The j_list and the l_list
# are both set to be [0] in the simulation.
# Input: 
#        J:                 (integer) the number of the modes.
#        p:                 (integer) the dimension of each mode.
#        r_U:               (integer) the prespecified rank for the common component U of each mode.
#        r_V:               (integer) the prespecified rank for the individual component V of each mode.
#        L:                 (integer) the number of tensors.
#        sigma:             (float) the noise level.
#        gap_seq:           (list) the distance between the minimum singular value of G_U and the maximum 
#                           singular value of G_V, where G_U and G_V are the two blocks of the core tensor
#                           G. An explicit definition is provided in our paper.
#        reps:              (integer) the number of repetitions. Defaults to be 100.
#        seed:              (integer) the random seed. Defaults to be 0.
# Output:
#       A tuple composed of three elements:
#       res_dist_all:       (numpy array) a matrix of size reps * 2, each row recording the estimation errors 
#                           for the common and individual components of the "distributed" method in each repeat.
#       res_single_all:     (numpy array) a matrix of size reps * 2, each row recording the estimation errors 
#                           for the common and individual components of the "single" method in each repeat.
#       res_pooled_all:     (numpy array) a vector of size reps recording the estimation error of the 
#                           "pooled" method in each repeat.

def simulation(J, p, r_U, r_V, L, sigma, gap_seq, reps=100, seed=0):

    np.random.seed(seed)

    res_dist_all = []
    res_pooled_all = []
    res_single_all = []

    r = r_U + r_V

    for gap in gap_seq:

        print(f"Begin simulations for SNR={gap}:")

        res_dist = []
        res_pooled = []
        res_single = []

        for rep in range(reps):

            if((rep + 1) % 10 == 0): 
                print(f"Iteration:{rep + 1}")
    
            # Generate U, V, G, T_star
            U_list = [np.linalg.qr(np.random.normal(size=(p, r_U)))[0] for _ in range(J)]
            V_list = [[np.linalg.qr((np.diag(np.ones(p)) - U @ np.transpose(U)) @ np.random.normal(size=(p, r_U)))[0] for U in U_list] for _ in range(L)]
            UV_list = [[np.hstack((U_list[j], V_list[l][j])) for j in range(J)] for l in range(L)]


            G_list = []
            for _ in range(L):
                GU = np.random.normal(size=tuple([r_U] * J))
                GU = GU * gap / singular_min(GU) 
            
                GV = np.random.normal(size=tuple([r_V] * J))
                GV = GV * (gap / 2) / singular_max(GV)

                G = np.zeros(tuple([r] * J))
                G[:r_U, :r_U, :r_U] = GU
                G[r_U:, r_U:, r_U:] = GV

                G_list.append(G)

            T_star_list = [tl.tucker_tensor.tucker_to_tensor((G_list[l], UV_list[l])) for l in range(L)]

            # Generate individual tensors
            Z_list = [np.random.normal(scale=sigma, size=tuple([p] * J)) for _ in range(L)]
            T_list = [T_star_list[l] + Z_list[l] for l in range(L)]     

            # Generate initial estimators
            UV_inits = [tl.decomposition.tucker(T, rank=tuple([r] * J))[1] for T in T_list] 

            # Distributed Tensor PCA
            U_est, V_est = dist_tensor_hetero(T_list, r_U, r_V, UV_inits, j_list=[0], l_list=[0])
            res_dist.append(np.array([rho(U_est[0], U_list[0]), rho(np.hstack((U_est[0], V_est[0][0])), np.hstack((U_list[0], V_list[0][0])))]))

            # Single machine for comparison

            U_single, V_single = dist_tensor_hetero([T_list[0]], r_U, r_V, [UV_inits[0]], j_list=[0], l_list=[0])
            res_single.append(np.array([rho(U_single[0], U_list[0]), rho(np.hstack((U_single[0], V_single[0][0])), np.hstack((U_list[0], V_list[0][0])))]))


            # Pooled tensors for comparison
            T_pooled = 0
            for T in T_list:
                T_pooled += (1/L) * T
            U_init_pooled = [tl.decomposition.tucker(T_pooled, rank=tuple([r] * J))[1]]
            U_est_pooled, _ = dist_tensor_hetero([T_pooled], r_U, r_V, U_init_pooled, j_list=[0])
            res_pooled.append(rho(U_est_pooled[0], U_list[0]))


            if((rep + 1) % 10 == 0):
                print(f"Results for rep {rep+1}:")
                print(f" distributed:")
                print(np.mean(np.array(res_dist), axis=0))
                print(f" single:")
                print(np.mean(np.array(res_single), axis=0))
                print(f" pooled:")
                print(np.mean(np.array(res_pooled)))


        res_dist_all.append(res_dist)
        res_pooled_all.append(res_pooled)
        res_single_all.append(res_single)
    
    return(np.array(res_dist_all), np.array(res_single_all), np.array(res_pooled_all))


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


gap_seq = [p ** gamma]

error_dist, error_single, error_pooled = simulation(J=3, p=p, r_U=3, r_V=3, L=L, sigma=1, 
                                                    gap_seq=gap_seq, reps=1000, seed=0)


# Stack the output into a matrix of size reps * 5
res_mat = np.hstack((error_dist.reshape(-1, 2), error_single.reshape(-1, 2), error_pooled.reshape(-1, 1)))

# Save the results
np.savetxt(f"results/hetero_2_res_p={p}_L={L}_gamma={gamma}.txt", res_mat)