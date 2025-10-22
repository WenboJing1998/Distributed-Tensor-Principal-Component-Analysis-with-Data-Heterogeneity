We provide the source codes for all of our numerical results, including two parts: the real data analysis and the simulations in the manuscript "Distributed Tensor Principal Component Analysis with Data Heterogeneity" (https://arxiv.org/abs/2405.11681)

The folder “real_data” provides self-contained Jupyter notebooks to record the workflow for the real data analysis. Specifically, running the Jupyter Notebook files “proteins.ipynb” and “PTC_FM_50.ipynb” will exactly reproduce Figures 4(a) and 4(b) in the manuscript, respectively. The corresponding datasets “PROTEINS.pt” and “PTC_FM_50_PI.pt” are in the same folder. The approximate run times for “PROTEINS.pt” and “PTC_FM_50_PI.pt” are 3-5 minutes and 10-15 minutes on a regular laptop device, respectively.
 
The folder “simulation” contains the source code for the simulations in the manuscript. Due to their long computation time, all the simulations run on NYU High-Performance Computing (HPC). HPC uses standard compute nodes with 192GB RAM and dual CPU sockets of 24-core Intel Cascade Lake Platinum 8268 chips. We provide the Python code files for the simulations and the sbatch files for running the Python files on NYU HPC. Specifically, 

(1) the “homo_hpc.py” file contains the source code of the simulations for the homogeneous setting, with the “tensor_homo.sbatch” file for submitting it to HPC (Figure 1 in the manuscript); 

(2) the “hetero_hpc_same_core.py” file contains the source code of the simulations for the heterogeneous setting with the same core tensors, with the “tensor_hetero_same_core.sbatch” file for submitting it to HPC (Figure 2 in the manuscript); 

(3) the “hetero_hpc_different_core.py” file contains the source code of the simulations for the heterogeneous setting with the different core tensors, with the “tensor_hetero_different_core.sbatch” file for submitting it to HPC (Figure 3 in the manuscript); 

The above three Python files save the results to a folder named “results.” After that, running “simulation_plot.ipynb” will reproduce Figures 1-3 in the manuscript. The approximate run times for “homo_hpc.py,” “hetero_hpc_same_core.py,” and “hetero_hpc_different_core.py” are 20-25 hours, 30-40 hours, and 30-40 hours on NYU HPC, respectively.
 
