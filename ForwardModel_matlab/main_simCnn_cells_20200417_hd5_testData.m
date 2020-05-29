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
% Note: Edit the content in the m file below
ExpSimCnn = SIM_EXPERIMENT_cnn2_cells('parameterFile_real20191229.m');parameterFile_real20191229

Nt_use = 32;
N_sls = 1;
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

%% make data store
imds = imageDatastore('./_data0_cells/Tst_data/');
I0_all_cells = imds.readall;

for i = 1:length(I0_all_cells)
    I0_all(:,:,i) = I0_all_cells{i};
end
% Y = img2blocks(I0_all,Nx); % ordered blocking functions
% Z = blocks2img(Y);
X0_all = img2blocks(I0_all,Nx);

%% run forward model for all images
saveDir = [sprintf('./_cnn_synthTstData/%d_sls_',N_sls) date '/']; 
mkdir(saveDir)

tic
Nb = size(I0_all,3)

for i = 1:size(X0_all,4)
    for j = 1:size(X0_all,5)
    
        X0 = X0_all(:,:,:,i,j);        
        X0 = single(X0);

        X0 = reshape(X0,[Nx*Nx Nb]);

        Y0 = A*X0;
        X0 = reshape(X0,Nx,Nx,Nb);
        Y0 = reshape(Y0,Nx,Nx,Nt_use,Nb);
        Y  = poissrnd(Y0);

        Y0 = Y0*ExpSimCnn.X0_gain+ExpSimCnn.X0_bg;
        Y0 = uint16(Y0);

        DataIn = Y0;
        DataGt = reshape(X0,Nx,Nx,1,Nb);

        % write to an HDF file
        fileNameStem = [saveDir sprintf('cells_i%d_j%d_tst_data_%dsls_',i,j,N_sls) date '.h5']
        hdf5_writeDataset(fileNameStem,DataIn,DataGt)

        display([size(DataIn) size(DataGt)])
        display(sprintf('\t[i j]=[%d %d]/%d ',i,j,size(X0_all,4)*size(X0_all,5)))
    end
end
toc






