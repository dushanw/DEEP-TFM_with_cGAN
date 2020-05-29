function exPSF = sim_get_modeled_exPSF(z,ls_ex,dx_gt)
    
    % excitaion resolution r_ex = 1.22 lambda/NA
    NA = 1;
    lambra = .8;% [um]
    r_ex =  1.22*lambra/NA;% [um] 
       
    % Inorm_ex_pk_z = exp(-2.5*z/ls_ex);% from [kim2007multifocal]
    Inorm_ex_pk_z = 1;% assume no loss at deep 
    
    fwhm_exPSF = r_ex;% [um]
    sigma_exPSF = fwhm_exPSF/2.35482;% in[um]
    sigma_exPSF = sigma_exPSF/dx_gt;% in PS pixel size

    exPSF_surf = fspecial('gaussian',ceil(sigma_exPSF*6),sigma_exPSF);
    exPSF = Inorm_ex_pk_z*exPSF_surf;

end