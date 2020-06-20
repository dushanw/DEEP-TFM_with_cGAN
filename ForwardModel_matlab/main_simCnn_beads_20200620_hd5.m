% 20181019 by Dushan N. Wadduwage
% 20190718 updated by Dushan N. Wadduwage
% 20191230 updated by Dushan N. Wadduwage
% 20200104 updated by Dushan N. Wadduwage
% 20200401 updated by Dushan N. Wadduwage to simulate cells data form Joe
% main.m

clc; close all; clear all;
addpath(genpath('./supportingFunctions/'))
addpath('./classes/')

%% setup experiment object
% Note: Edit the content in the m file below for experiment parameters
ExpSimCnn = SIM_EXPERIMENT_cnn2_cells('parameterFile_real20191229.m');parameterFile_real20191229

Nt_use = 32;
N_sls = 5;
z = sl_em*N_sls;
maxCounts = 200;

ExpSimCnn.build_A_efficient(Nt_use,ny,nx,z);  
A = ExpSimCnn.A;

%% run forward model for all images
saveDir = ['./_cnn_synthTrData/' date '/']; 
mkdir(saveDir)

tic
t=1;
for kk = 1:200
    X0_sig = 1000;
    MiniBatchSize = 128;
    parfor i=1:MiniBatchSize
      X0(:,:,i) = gen_beadsIn2D(Nx,Ny,0)*X0_sig;% range [0 1]
    end
    
    X0 = reshape(X0,[Nx*Nx MiniBatchSize]);
    X0 = single(X0);

    Y0 = A*X0;
    X0 = reshape(X0,Nx,Nx,MiniBatchSize);
    Y0 = reshape(Y0,Nx,Nx,Nt_use,MiniBatchSize);
    Y  = poissrnd(Y0);

    Y = Y*ExpSimCnn.X0_gain;
    Y = uint16(Y);

    DataIn(:,:,:,t:t+MiniBatchSize-1) = Y;
    DataGt(:,:,:,t:t+MiniBatchSize-1) = reshape(X0,Nx,Nx,1,MiniBatchSize);
    
    t=t+MiniBatchSize;
    display(sprintf('\tkk=%d/%d \t| t=%d',kk,floor(patchds.NumObservations/patchds.MiniBatchSize),t))
end
toc

% write to an HDF file
fileNameStem = [saveDir sprintf('beads_tr_data_%dsls_',N_sls) date '.h5']
hdf5_writeDataset(fileNameStem,DataIn,DataGt)

% test read
% dataGt_rd = h5read(fileNameStem,'/input');




