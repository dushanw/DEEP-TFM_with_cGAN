% notes: 
%   The parameters: NA 1.0; Size of camera pixel on sample 330nm; exc/em wavelength: 800nm/590nm. [Cheng 2020-12-24]


function pram = f_pram_init()

  %% names
  pram.mic_typ      = 'DMD';                            % {'DMD','WGD'}
  pram.pattern_typ  = 'dmd_exp_tfm_beads_8';            % {'dmd_sim_rnd',
                                                        %  'dmd_exp',
                                                        %  'dmd_exp_tfm',
                                                        %  'dmd_exp_tfm_beads_3',
                                                        %  'dmd_exp_tfm_beads_4',
                                                        %  'dmd_exp_tfm_beads_8',
                                                        %  'wgd_sim',
                                                        %  'wgd_exp'
                                                        % }
  pram.dataset      = 'minist';                         % {'minist',
                                                        %  'andrewCells_fociW3_63x_maxProj',
                                                        %  'andrewCells_dapi_20x_maxProj',}
  pram.psf_typ      = 'MC';                             % {'MC','gaussian',...}
  
  %% data size parameters
  pram.Nx      = 64;
  pram.Ny      = 64;
  pram.Nz      = 64;
  pram.Nc      = 1;
  pram.Nt      = 128;
  pram.dx      = 0.33;                                  % data pixel size
  
  %% MIC and imaging parameters
  pram.lambda_ex  = 0.800;                              % [um]      excitation wavelength
  pram.lambda_em  = 0.590;                              % [um]      emission wavelength {0.606 }
  pram.NA         = 1;                                  % [AU]      numerical aperture of the objective
  pram.z0_um      = -50;                                % [um]  
  
  %% optical properties of the tissue
  pram.mus        = 200;                                % [cm^-1]   scattering coefficient of tissue
  pram.g          = 0.90;                               % [AU]      anisotropy of scattering of tissue
  pram.nt         = 1.33;                               % [AU]      refractive index of tissue  
  pram.nm         = 1.33;                               % [AU]      refractive index of the medium (ex:water,air)  
  pram.sl         = (1/pram.mus)*10*1e3;                % [um]      sacttering length  
  
  %% camera parameters <THIS IS OLD NEED UPDATING WITH THE NEW CAM MODEL>
  pram.amp     = 1e8;                                   % scaling factor from measured images to images in [0 1]
  pram.mu_rd   = 100;
  pram.sd_rd   = 10;
  pram.binR    = 1;
  
end