% notes: 
%   The parameters: NA 1.0; Size of camera pixel on sample 330nm; exc/em wavelength: 800nm/590nm. [Cheng 2020-12-24]


function pram = f_pram_init_invMatSolv()

  %% names
  pram.mic_typ      = 'DMD';                            % {'DMD','WGD'}
  pram.pattern_typ  = 'dmd_sim_rnd';                    % {'dmd_sim_rnd',
                                                        %  'dmd_exp_tfm_beads_7sls_20201219'   
                                                        % }
  pram.dataset      = 'beads';                          % {'minist',
                                                        %  'andrewCells_fociW3_63x_maxProj',
                                                        %  'andrewCells_dapi_20x_maxProj',
                                                        %  'beads'}
  pram.psf_typ      = 'MC';                             % {'MC','gaussian',...}
    
  %% optical properties of the tissue
  pram.mus        = 200;                                % [cm^-1]   scattering coefficient of tissue
  pram.g          = 0.90;                               % [AU]      anisotropy of scattering of tissue
  pram.nt         = 1.33;                               % [AU]      refractive index of tissue  
  pram.nm         = 1.33;                               % [AU]      refractive index of the medium (ex:water,air)  
  pram.sl         = (1/pram.mus)*10*1e3;                % [um]      sacttering length  

  %% data size parameters
  pram.Nx      = 16;
  pram.Ny      = 16;
  pram.Nz      = 64;
  pram.Nc      = 1;
  pram.Nt      = pram.Nx * pram.Ny;
  pram.Nb      = 1e4;                                   %           number of batches (instances)
% pram.dx      = 0.33*4+0.33/4;                         % [um]      0.33 for DEEP-TFM Nuvu cam;
  pram.dx      = 0.33*4;                                % [um]      0.33 for DEEP-TFM Nuvu cam;
  pram.dz      = 1;
  
  %% MIC and imaging parameters
  pram.lambda_ex  = 0.800;                              % [um]      excitation wavelength
  pram.lambda_em  = 0.590;                              % [um]      emission wavelength {0.606 }
  pram.NA         = 1;                                  % [AU]      numerical aperture of the objective
  pram.z0_um      = -7*pram.sl;                         % [um]      depth (z=0 is the surface and -ve is below)
    
  %% camera parameters <THIS IS OLD NEED UPDATING WITH THE NEW CAM MODEL>
  pram.binR    = 1;
  pram.cam_emhist_Nreps = 10000;
  
  %% run environment parameters  
  pram.useGPU  = gpuDeviceCount>1 ;
  
end




