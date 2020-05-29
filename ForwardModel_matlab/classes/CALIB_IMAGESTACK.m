% 20181019 by Dushan N. Wadduwage
% Calibration images class definition

classdef CALIB_IMAGESTACK < IMAGESTACK
   methods  
       % preprocess calibration image set
       function self = preprocess(self) 
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           for i=1:self.Ny
               for j=1:self.Nx                    
                     self.Data(i,j,:)= (self.Data(i,j,:) - ...
                        mean(squeeze(self.Data(i,j,:))))/sum(squeeze(self.Data(i,j,:)));% real data
                        
%                      self.Data(i,j,:)= 2*(self.Data(i,j,:) - mean(squeeze(self.Data(i,j,:))));% simulated Data  
                    
%                     self.Data(i,j,:)= (self.Data(i,j,:) - mean(squeeze(self.Data(i,j,:))));
%                     self.Data(i,j,:)= self.Data(i,j,:)./sqrt(sum(squeeze(self.Data(i,j,:)).^2));
               end
           end
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
