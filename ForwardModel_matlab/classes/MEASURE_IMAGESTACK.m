% 20181019 by Dushan N. Wadduwage
% Measurement images class definition

classdef MEASURE_IMAGESTACK < IMAGESTACK
   methods          
       
       function self = preprocess(self)  
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           tempMean = mean(self.Data,3);
           for i=1:self.Nt                
               self.Data(:,:,i) = (self.Data(:,:,i)-tempMean);% working for real data
%               self.Data(:,:,i) = (2*self.Data(:,:,i)-tempMean);% test for simulated data               
           end
       end
       
       function self = addPoissonNoise(self)                                             
           self.Data(:) = poissrnd(self.Data(:));
       end
       
       function self = denoise(self)
           net = denoisingNetwork('dncnn');
           mx = max(self.Data(:));
           for i=1:self.Nt
               In = self.Data(:,:,i);               
               In = In/mx;
               res = net.activations(In, 59,'OutputAs','channels');                       
               self.Data(:,:,i) = In-res;
           end
       end
       
   end
end
