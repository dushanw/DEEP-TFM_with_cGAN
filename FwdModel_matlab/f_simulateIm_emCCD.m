function [Xhat XhatADU] = f_simulateIm_emCCD(X0,pram)
  
  Xdark   = pram.cam_dXdt_dark * pram.cam_t_exp;     % [e-]    Dark noise
  Xhat    = poissrnd(X0 + Xdark);                    % [e-]    Poisson shot noise  

  if pram.useGPU ==1
    Xhat  = gpuArray(Xhat);
  end
  
%   x0_in   = [1:max(Xhat(:))]';
%   x0_out  = repmat(x0_in,[1 10000]);
%   fprintf('%0.4d/%0.4d',0,pram.cam_N_gainStages)
%   for i=1:pram.cam_N_gainStages
%     fprintf('\b\b\b\b\b\b\b\b\b%0.4d/%0.4d',i,pram.cam_N_gainStages)
%     x0_out  = x0_out + binornd(x0_out,pram.cam_Brnuli_alpha);               % [e-]    Add binomial noise due to each step in the Em process
%   end
%   fprintf('\n')
  
  fprintf('%0.4d/%0.4d',0,pram.cam_N_gainStages)
  for i=1:pram.cam_N_gainStages 
    fprintf('\b\b\b\b\b\b\b\b\b%0.4d/%0.4d',i,pram.cam_N_gainStages)
    Xhat  = Xhat + binornd(Xhat,pram.cam_Brnuli_alpha);               % [e-]    Add binomial noise due to each step in the Em process
  end
  fprintf('\n')
  Xhat    = (Xhat + normrnd(0,pram.cam_sigma_rd))/pram.cam_EMgain;        % [e-]    Input (to EM register) image with noise
  XhatADU = Xhat * pram.cam_EMgain * pram.cam_ADCfactor + pram.cam_bias;      % [ADU]   simulated image in ADU
end