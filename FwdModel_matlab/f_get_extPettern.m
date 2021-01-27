
function [E Y_exp X_refs pram] = f_get_extPettern(pram)

  switch pram.pattern_typ
    case 'dmd_sim_rnd'
      E     = single(rand([pram.Ny pram.Nx pram.Nt])>0.5); % for DMDs
      Y_exp = [];
      X_refs= [];
      pram  = pram;
    case 'dmd_sim_Hadamard'
      H     = hadamard(pram.Ny*pram.Nx);
      A     = (H+1)/2;
      E     = single(reshape(A,pram.Ny,pram.Nx,[])); % for DMDs
      E     = E(:,:,1:pram.Nt);
      
      Y_exp = [];
      X_refs= [];
      pram  = pram;
    case 'dmd_exp_tfm_beads_7sls_20201219'
      load('./_extPatternsets/dmd_exp_tfm_beads_7sls_20201219.mat')
      
      y_inds  = size(Data.Ex,1)-pram.Ny+1:size(Data.Ex,1);      % select the lower left coner as it's brighter
      x_inds  = 1:pram.Nx;
      t_inds  = 20 + (1:pram.Nt);
      
      E                   = single(Data.Ex           (y_inds,x_inds,:));
      Y_exp.beads1        = single(Data.beads1_7sls  (y_inds,x_inds,:));
      Y_exp.beads2        = single(Data.beads2_7sls  (y_inds,x_inds,:));
      X_refs.beads1_sf_wf0= single(Data.beads1_sf_wf0(y_inds,x_inds,:));
      X_refs.beads2_sf_wf0= single(Data.beads2_sf_wf0(y_inds,x_inds,:));

      X_refs.beads1_avg0  = mean(Y_exp.beads1,3);
      X_refs.beads2_avg0  = mean(Y_exp.beads2,3);
      
      E                   = E           (:,:,t_inds);
      Y_exp.beads1        = Y_exp.beads1(:,:,t_inds);
      Y_exp.beads2        = Y_exp.beads2(:,:,t_inds);
      
      X_refs.beads1_avg   = mean(Y_exp.beads1,3);
      X_refs.beads2_avg   = mean(Y_exp.beads2,3);
      
      % normalize E
      E                   = E - min(E(:));
      E                   = E / max(E(:));
      
      pram.maxcount         = max([X_refs.beads1_avg0(:); X_refs.beads2_avg0(:)]);
      pram.dx               = Data.pram_ex.dx0;
      pram.cam_bias         = Data.pram_beads.bias;
      pram.cam_ADCfactor    = Data.pram_beads.ADCfactor;
      pram.cam_EMgain       = Data.pram_beads.EMgain;
      pram.cam_t_exp        = Data.pram_beads.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd     = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark    = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                              %             Number of Em-gain stages      
  end

end


function H = subf_get_hadamard_patterns(Ny,Nx,Nt)

  ny      = sqrt(Nt/cos(pi/6));                     % from tesselation nx = ny*cos(pi/6) and Area = Nt = nx*ny  
  ny      = round(ny);
  nx      = floor(Nt/ny);

  hadmat  = hadamard(Nt);
% hadmat(:,1)  = [1:Nt]';                           % to check the that the tesselation is right
  
  h       = reshape(hadmat(1:nx*ny,:),[ny nx Nt]);  % one block
  h2      = [h circshift(h,round(size(h,1)/2),1)];  % two blocks for triangulization

  H       = repmat(h2,[ceil(Ny/ny) ceil(Nx/(2*nx)) 1]);
  H       = H(1:Ny,1:Nx,:);
end

