
pram = f_pram_init();
PSFs = f_simPSFs(pram);

save(['./_PSFs/' datestr(datetime('now'),'yyyy-mm-dd') '_PSFs.mat'],'PSFs')