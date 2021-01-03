
function emhist = f_genEmhist(max_input_photons,N_reps,pram)

  in_photons  = [1:max_input_photons]';

  emhist_100  = (repmat(in_photons,[1 100]));
  for i=1:N_reps/100
    emhist{i} = emhist_100;
  end
   
%   if pram.useGPU ==1
%     emhist  = gpuArray(emhist);
%   end

  tic
  %  r = zeros(size(x0_out));
  fprintf('%0.4d/%0.4d',0,pram.cam_N_gainStages)
  parfor tt=1:length(emhist)
    for ii=1:pram.cam_N_gainStages
      fprintf('\b\b\b\b\b\b\b\b\b%0.4d/%0.4d',ii,pram.cam_N_gainStages)     
      emhist{tt}  = emhist{tt} + binornd(emhist{tt},pram.cam_Brnuli_alpha);
  %     r(:)=0;
  %     for i = 1:max(x0_out(:))    
  %       k = find(x0_out >= i);      
  %       r(k) = r(k) + (rand(size(k)) < pram.cam_Brnuli_alpha);
  %     end    
  %     x0_out  = x0_out + r;
    end
  end
  fprintf('\n')
  toc

  emhist = [emhist{:}];
  
  mkdir('./_emhist')
  save(['./_emhist/emhist_' datestr(datetime('now')) '.mat'],'emhist')  
end


