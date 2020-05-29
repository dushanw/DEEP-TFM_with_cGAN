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
N_sls = 6;
z = sl_em*N_sls;
maxCounts = 200;

ExpSimCnn.build_A_efficient(Nt_use,ny,nx,z);  
A = ExpSimCnn.A;

%% generate initial training and testing images from original SOM_mouse images
% Noise_th = 10;
% filepath = '/home/harvard/Dropbox (Harvard University)/WorkingData/20190614_SOM/20190317_SOM mice';
% imds = imageDatastore(filepath,'ReadFcn',@(dataPath) readImages_cell(dataPath,Noise_th),'IncludeSubfolders',true,'FileExtensions',{'.tif','.tiff'});
% X0_all = imds.readall;
% writeToDataFolder(X0_all);

%% make data store of images
imds = imageDatastore('./_data0_cells/Tr_data/');
patchds = randomPatchExtractionDatastore(imds,imds,[128 128],'PatchesPerImage',64);
patchds.MiniBatchSize = 2^10;                                                       % change this value to fit GPU memory

%% run forward model for all images
saveDir = ['./_cnn_synthTrData/' date '/']; 
mkdir(saveDir)

tic
t=1;
for kk = 1%:floor(patchds.NumObservations/patchds.MiniBatchSize)
    
    X0_in = read(patchds);
    X0 = cell2mat(X0_in.InputImage);
    X0 = reshape(X0,[Nx patchds.MiniBatchSize Nx]);
    X0 = permute(X0,[1 3 2]);

    X0 = reshape(X0,[Nx*Nx patchds.MiniBatchSize]);
    X0 = single(X0);

    idx= find(max(X0,[],1)>20);
    Nbatch_now = length(idx);
    X0 = X0(:,idx);  

    Y0 = A*X0;
    X0 = reshape(X0,Nx,Nx,Nbatch_now);
    Y0 = reshape(Y0,Nx,Nx,Nt_use,Nbatch_now);
    Y  = poissrnd(Y0);

    Y0 = Y0*ExpSimCnn.X0_gain+ExpSimCnn.X0_bg;
    Y0 = uint16(Y0);

    DataIn(:,:,:,t:t+Nbatch_now-1) = Y0;
    DataGt(:,:,:,t:t+Nbatch_now-1) = reshape(X0,Nx,Nx,1,Nbatch_now);
    
    t=t+Nbatch_now;
    display([size(DataIn) size(DataGt)])
    display(sprintf('\tkk=%d/%d \t| t=%d',kk,floor(patchds.NumObservations/patchds.MiniBatchSize),t))
end
toc

% write to an HDF file
fileNameStem = [saveDir sprintf('cells_tr_data_%dsls_',N_sls) date '.h5']
hdf5_writeDataset(fileNameStem,DataIn,DataGt)

% test read
dataGt_rd = h5read(fileNameStem,'/input');




