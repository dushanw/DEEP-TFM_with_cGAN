% 20201122 by Dushan N. Wadduwage
% main forward model for cGAN-DEEP-TFM


pram              = f_pram_init();

%% read exp data
[E Y_exp X_refs]  = f_get_extPettern(pram);

%% simulate training data  
N_beads           = 10;
X0                = f_genobj_beads3D(pram.Ny,pram.Nx,pram.Nz,N_beads);

