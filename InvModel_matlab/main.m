% 20200629 by Dushan N. Wadduwage

clc; close all; clear all;

% read training read
%fileName  = '../ForwardModel_matlab/_cnn_synthTrData/20-Jun-2020/beads_tr_data_5sls_20-Jun-2020.h5';
%namestem  = 'net_sig1000counts'
% fileName  = '../ForwardModel_matlab/_cnn_synthTrData/30-Jun-2020_mc200_withBg/beads_tr_data_5sls_30-Jun-2020.h5';
% namestem  = 'net_sig200counts_withBg';
fileName  = '../ForwardModel_matlab/_cnn_synthTrData/30-Jun-2020_mc200_withBg2/beads_tr_data_5sls_30-Jun-2020.h5';
namestem  = 'net_sig200counts_withBg2';

X         = h5read(fileName,'/input');
Y         = h5read(fileName,'/gt');

% normalize data
X         = single(X)/2000;
Y         = single(Y)/100;

size_In   = size(X);
lgraph    = gennet_dncnnImgTranslator(size_In(1:3));
  
pram.numEpochs              = 100;    
pram.learnRateDropFactor    = 0.1;
pram.learnRateDropInterval  = 25;
pram.learnRateImgprocessor  = 1;% 0.0002;
pram.miniBatchSize          = 64; 

% net       = tr_net_dirct(lgraph,X(:,:,:,1:64*64),Y(:,:,:,1:64*64),pram);
net       = tr_net_dirct(lgraph,X,Y,pram);
lgraph    = layerGraph(net);
save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'net');

%load(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'net');
XTest     = X(:,:,:,end);
YTestGt   = Y(:,:,:,end); 
YTestHat  = predict(net,XTest);

%% Test on the experimental dataset
load('../ForwardModel_matlab/crop_rect_20191229_50_fgbg64.mat')
YrealGt   = imread('../ForwardModel_matlab/_data_patterns/beads_0_5_50_gt/beads_water_wf.tif');
YrealGt   = imresize(imcrop(YrealGt,crop_rect{1}),[size_In(1) size_In(2)]);
load('../ForwardModel_matlab/Yreal0.mat');
Xreal     = Data/2000;

YRealHat  = predict(net,Xreal);
imagesc([rescale(XTest(:,:,1)) rescale(YTestHat) rescale(YTestGt);...
         rescale(Xreal(:,:,1)) rescale(YRealHat) rescale(YrealGt)...
         ]);axis image
saveas(gca,['./_figs/' date '_' namestem '.tif']);




