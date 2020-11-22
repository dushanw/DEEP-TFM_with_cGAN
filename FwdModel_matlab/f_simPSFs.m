
function PSFs = f_simPSFs(pram)
  of        = cd('./_submodules/MC_LightScattering/');

  %% set prams
  mclm_pram           = f_praminit();
  mclm_pram.savepath  = [of '/_PSFs/'];
  mclm_pram.fNameStem = 'MC';
  mclm_pram.Nx        = pram.Nx*4;
  mclm_pram.dx        = pram.dx;
  mclm_pram.z0_um     = pram.z0_um;
  mclm_pram.z0_um     = pram.z0_um;
  mclm_pram.sl        = pram.sl;
  mclm_pram.mus       = pram.mus;
  mclm_pram.lambda_ex = pram.lambda_ex;
  mclm_pram.lambda_em = pram.lambda_em;
  mclm_pram.NA        = pram.NA;
  mclm_pram.Nphotons  = 1E7;
  mclm_pram.Nsims     = 16;
  mclm_pram.useGpu    = 1;

  %% simulate sPSF (saves to [mclm_pram.savepath mclm_pram.fNameStem '_sPSF.mat'])
  main(mclm_pram);
  load([mclm_pram.savepath mclm_pram.fNameStem '_sPSF.mat']);       % loads sPSF
  sPSF  = sPSF(2:end-1,2:end-1);
  
  %% simulate exPSF and emPSF
  cd('/optical_PSF/');
  exPSF     = Efficient_PSF(mclm_pram.NA, mclm_pram.nm, mclm_pram.lambda_ex, mclm_pram.dx,mclm_pram.Nx,mclm_pram.Nx,2,200);
  exPSF     = exPSF(:,:,2);
  emPSF     = Efficient_PSF(mclm_pram.NA, mclm_pram.nm, mclm_pram.lambda_em, mclm_pram.dx,mclm_pram.Nx,mclm_pram.Nx,2,200);
  emPSF     = emPSF(:,:,2);
  cd(of);
  
  PSFs.exPSF = exPSF;
  PSFs.emPSF = emPSF;
  PSFs.sPSF  = sPSF;
  
  save([mclm_pram.savepath 'PSFs.mat'],'PSFs');       % loads sPSF
end