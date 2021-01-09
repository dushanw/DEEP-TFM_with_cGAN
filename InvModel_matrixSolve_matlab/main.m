% 20210107 by Dushan N. Wadduwage (wadduwage@fas.harvard.edu)
% main matrix based inverse model for DEEP-TFM

% 20210107 : used to compare PS, SPx, and DEEP

clc;clear all;close all
addpath('../FwdModel_matlab/')
addpath('../FwdModel_matlab/_extPatternsets/')

pram = f_pram_init_invMatSolv();
Nx_list = [8 16 32 64];
Ny_list = [8 16 32 64];
    
for ii=1:length(Nx_list)  
  pram.Nx = Nx_list(ii);
  pram.Ny = Ny_list(ii);
  pram.Nt = pram.Nx * pram.Ny;
  
  [E Y_exp X_refs pram] = f_get_extPettern(pram);

  %% simulate sPSF, exPSF, and emPSF
  of = cd('../FwdModel_matlab/');
  PSFs    = f_simPSFs(pram);
  cd(of)
  %load('../FwdModel_matlab/_PSFs/PSFs27-Dec-2020 04_21_23.mat')        % on Macbook

  %% simulate data  
  % N_beads = 2 * round(500 * (pram.Nx*pram.Ny*pram.Nz*pram.dx^2*pram.dz)/(256*256*64*0.33^2*1));
  % X0      = f_genobj_beads3D_1um_4um(N_beads,pram);  
  % ind     = find(sum(sum(X0,1),2)>0);
  % X0      = X0(:,:,ind(randi(length(ind))));
  % X0      = X0 - min(X0(:));
  % X0      = X0/max(X0(:));

  % imagesc(X0);axis image;colorbar

  % MINST
  load('../FwdModel_matlab/_datasets/mnist.mat')
  X0      = imresize(XTest(:,:,1,3),[pram.Ny pram.Nx]);
  X0      = X0 - min(X0(:));
  X0      = X0/max(X0(:));

  %% Sensing matrices 
  [A A_spx A_ps]= f_gen_fwdA(E,PSFs,pram);
  At            = inv(A'*A)*A';
  At_spx        = inv(A_spx);
  At_ps         = inv(A_ps);

  %% run simulation
  clear Xhat_ps Xhat_deep Xhat_spx X_gt
  for i = 1:5
    X             = X0*10^(i-1);
    % sumnerical simulation
    for j=1:100
      [i j]
    %  Xhat_ps  (:,:,i) = reshape(At_ps *poissrnd(A_ps *X(:)),[pram.Ny pram.Nx]);    
      Xhat_ps  (:,:,j,i) = reshape(       poissrnd(A_ps *X(:)),[pram.Ny pram.Nx]);  
      Xhat_deep(:,:,j,i) = reshape(At    *poissrnd(A    *X(:)),[pram.Ny pram.Nx]);
      Xhat_spx (:,:,j,i) = reshape(At_spx*poissrnd(A_spx*X(:)),[pram.Ny pram.Nx]);  
    end
    X_gt(:,:,1,i) = X;
  end
  save(sprintf('sim_sls-%d_NyNx-%dx%d.mat',-PSFs.pram.z0_um/PSFs.pram.sl,...
                                           pram.Ny,...
                                           pram.Nx),...
       'Xhat_ps','Xhat_deep','Xhat_spx','X_gt','pram','PSFs');   
end

   
%% temp analysis code   
   
% figure;
% subplot(2,2,1);imagesc(imtile(X_gt     (:,:,1,:)));axis image;colorbar;title('GT  ')
% subplot(2,2,2);imagesc(imtile(Xhat_ps  (:,:,1,:)));axis image;colorbar;title('PS  ')
% subplot(2,2,3);imagesc(imtile(Xhat_spx (:,:,1,:)));axis image;colorbar;title('SPX ')
% subplot(2,2,4);imagesc(imtile(Xhat_deep(:,:,1,:)));axis image;colorbar;title('DEEP')
% 
% 
% sigma_Xhat_ps   = std(Xhat_ps  ,0,3);
% sigma_Xhat_deep = std(Xhat_deep,0,3);
% sigma_Xhat_spx  = std(Xhat_spx ,0,3);
% 
% snr_deep        = X_gt./sigma_Xhat_deep;
% snr_spx         = X_gt./sigma_Xhat_spx;
% snr_ps          = X_gt./sigma_Xhat_ps;
% 
% snr_ps(isnan(snr_ps))=0;
% 
% figure;
% subplot(2,2,1);imagesc(imtile(X_gt))     ;axis image;colorbar;title('GT  ')
% subplot(2,2,2);imagesc(imtile(snr_ps))   ;axis image;colorbar;title('PS  ')
% subplot(2,2,3);imagesc(imtile(snr_spx))  ;axis image;colorbar;title('SPX ')
% subplot(2,2,4);imagesc(imtile(snr_deep)) ;axis image;colorbar;title('DEEP')
% 
% semilogy(squeeze(mean(mean(snr_ps,1),2)));hold on
% semilogy(squeeze(mean(mean(snr_spx,1),2)));
% semilogy(squeeze(mean(mean(snr_deep,1),2)));hold off
% legend({'ps','spx','deep'})

% example Y
% Y_ps          = reshape(A_ps *X(:),pram.Ny,pram.Nx,[]);
% Y_deep        = reshape(A    *X(:),pram.Ny,pram.Nx,[]);
% Y_spx         = reshape(A_spx*X(:),pram.Ny,pram.Nx,[]);
% %imagesc([rescale(Y_ps(:,:,1)) rescale(Y_deep(:,:,1)) rescale(Y_ps(:,:,1))]);axis image
% subplot(2,2,1);imagesc(X(:,:,1))     ;axis image;colorbar
% subplot(2,2,2);imagesc(Y_ps(:,:,1))  ;axis image;colorbar
% subplot(2,2,3);imagesc(Y_spx(:,:,1)) ;axis image;colorbar
% subplot(2,2,4);imagesc(Y_deep(:,:,1));axis image;colorbar

% Xhat_ps  (:,:,1) = reshape(At_ps *(A_ps *X(:)),[pram.Ny pram.Nx]);
% Xhat_deep(:,:,1) = reshape(At    *(A    *X(:)),[pram.Ny pram.Nx]);
% Xhat_spx (:,:,1) = reshape(At_spx*(A_spx*X(:)),[pram.Ny pram.Nx]);  

% figure;
% subplot(2,2,1);imagesc(X        (:,:,1));axis image;colorbar;title('GT  ')
% subplot(2,2,2);imagesc(Xhat_ps  (:,:,1));axis image;colorbar;title('PS  ')
% subplot(2,2,3);imagesc(Xhat_spx (:,:,1));axis image;colorbar;title('SPX ')
% subplot(2,2,4);imagesc(Xhat_deep(:,:,1));axis image;colorbar;title('DEEP')
% 
% figure;
% subplot(2,2,1);imagesc(mean(X        ,3));axis image;colorbar;title('GT  ')
% subplot(2,2,2);imagesc(mean(Xhat_ps  ,3));axis image;colorbar;title('PS  ')
% subplot(2,2,3);imagesc(mean(Xhat_spx ,3));axis image;colorbar;title('SPX ')
% subplot(2,2,4);imagesc(mean(Xhat_deep,3));axis image;colorbar;title('DEEP')
% 
% sigma_Xhat_ps   = std(Xhat_ps  ,0,3);
% sigma_Xhat_deep = std(Xhat_deep,0,3);
% sigma_Xhat_spx  = std(Xhat_spx ,0,3);
% 
% snr_deep        = X./sigma_Xhat_deep;
% snr_spx         = X./sigma_Xhat_spx;
% snr_ps          = X./sigma_Xhat_ps;
% 
% figure;
% subplot(2,2,1);imagesc(X)        ;axis image;colorbar
% subplot(2,2,2);imagesc(snr_ps)   ;axis image;colorbar
% subplot(2,2,3);imagesc(snr_spx)  ;axis image;colorbar
% subplot(2,2,4);imagesc(snr_deep) ;axis image;colorbar
% 

% sigma_Xhat_deep = reshape(sqrt(At.^2*A*X(:)),[pram.Ny pram.Nx]);
% sigma_Xhat_spx  = reshape(sqrt(At_spx.^2*A_spx*X(:)),[pram.Ny pram.Nx]);
% %sigma_Xhat_ps   = reshape(sqrt(At_ps.^2*A_ps*X(:)),[pram.Ny pram.Nx]);
% sigma_Xhat_ps   = reshape(sqrt(          A_ps*X(:)),[pram.Ny pram.Nx]);
% 
% snr_deep  = X./sigma_Xhat_deep;
% snr_spx   = X./sigma_Xhat_spx;
% snr_ps    = X./sigma_Xhat_ps;
% 
% % snr_gt    = sqrt(X);
% figure;
% subplot(2,2,1);imagesc(X)           ;axis image;colorbar;title('GT  ')
% subplot(2,2,2);imagesc(snr_ps)      ;axis image;colorbar;title('PS  ')
% subplot(2,2,3);imagesc(snr_spx)     ;axis image;colorbar;title('SPX ')
% subplot(2,2,4);imagesc(snr_deep)    ;axis image;colorbar;title('DEEP')
% 
% <next validate the snr values using a numerical simulation of the fwd model>
%     - there are some complications with the expsf for point scan. Try lower resolution  
% <if the two agree, write down the method> 
%     - good to write down the method


