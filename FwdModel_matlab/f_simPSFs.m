
function PSFs = f_simPSFs(pram)
  of        = cd('./_submodules/MC_LightScattering/');
  
  %% set prams
  mcls_pram           = f_praminit();
  mcls_pram.savepath  = [of '/_PSFs/'];
  mcls_pram.fNameStem = 'MC';
  mcls_pram.Nx        = pram.Nx*4+3;
  mcls_pram.dx        = pram.dx/2;
  mcls_pram.z0_um     = pram.z0_um;
  mcls_pram.z0_um     = pram.z0_um;
  mcls_pram.sl        = pram.sl;
  mcls_pram.mus       = pram.mus;
  mcls_pram.lambda_ex = pram.lambda_ex;
  mcls_pram.lambda_em = pram.lambda_em;
  mcls_pram.NA        = pram.NA;
  mcls_pram.Nphotons  = 1E7;
  mcls_pram.Nsims     = 16;
  mcls_pram.useGpu    = 1;
  
  %% simulate sPSF (saves to [mclm_pram.savepath mclm_pram.fNameStem '_sPSF.mat'])
  main(mcls_pram);
  load([mcls_pram.savepath mcls_pram.fNameStem '_sPSF.mat']);       % loads sPSF
  sPSF  = sPSF(2:end-1,2:end-1);
  
  %% simulate exPSF and emPSF
  cd('_supToolboxes/optical_PSF/');
  APSF_3D     = Efficient_PSF(mcls_pram.NA, mcls_pram.nm, mcls_pram.lambda_ex, mcls_pram.dx,mcls_pram.Nx-2,mcls_pram.Nx-2,2,200);
  PSF_3D      = abs(APSF_3D{1}).^2+abs(APSF_3D{2}).^2+abs(APSF_3D{3}).^2;
  exPSF       = PSF_3D(:,:,2);
  exPSF       = exPSF/sum(exPSF(:));
  
  APSF_3D     = Efficient_PSF(mcls_pram.NA, mcls_pram.nm, mcls_pram.lambda_em, mcls_pram.dx,mcls_pram.Nx-2,mcls_pram.Nx-2,2,200);
  PSF_3D      = abs(APSF_3D{1}).^2+abs(APSF_3D{2}).^2+abs(APSF_3D{3}).^2;
  emPSF       = PSF_3D(:,:,2);
  emPSF       = emPSF/sum(emPSF(:));
  
  cd(of);
  
  PSFs.exPSF  = exPSF;
  PSFs.emPSF  = emPSF;
  PSFs.sPSF   = sPSF;
  PSFs.pram   = mcls_pram;
  
  save([mcls_pram.savepath 'PSFs' datestr(datetime('now')) '.mat'],'PSFs'); % save sPSF
  
end
