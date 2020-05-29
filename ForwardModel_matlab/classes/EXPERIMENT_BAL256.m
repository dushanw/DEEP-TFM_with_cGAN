% 20181022 by Dushan N. Wadduwage
% data class definition

classdef EXPERIMENT_BAL256 < handle
   properties
       Y
       Ex
       X0
       Xhat
       sPSF
   end
   methods
       % constructor
       function obj = EXPERIMENT_BAL256(parameterFile,Nt)
           run(parameterFile);
                      
           Ex = CALIB_IMAGESTACK([],extPath,extNameStem,Nx,Ny,256,EmGain,crop_rect_name_ex); 
           Y = MEASURE_IMAGESTACK([],dataPath,dataNameStem,Nx,Ny,256,EmGain,crop_rect_name);
           
           obj.Ex = CALIB_IMAGESTACK(Ex.Data(:,:,[1:Nt/2 128:128+Nt/2-1]));
           obj.Y = MEASURE_IMAGESTACK(Y.Data(:,:,[1:Nt/2 128:128+Nt/2-1]));
           
           % obj.sPSF = PSF([],Nx,Ny);
           obj.Y.denoise;
           obj.Ex.denoise;
           
           obj.X0 = REC_IMAGESTACK(mean(obj.Y.Data,3));
           obj.Xhat = REC_IMAGESTACK(mean(obj.Y.Data,3));
           
           % preprocess data
           obj.Ex.preprocess;
           obj.Ex.diagonalize;
           
%           obj.Y.preprocess;           
       end
       
       function self = estimate_Xhat0(self)
               Y_temp = MEASURE_IMAGESTACK(self.Y.Data);           
               Y_temp.Data(find(Y_temp.Data(:)<0))=0;
                      
               Y_temp.preprocess;           
               Y_temp.columnize;
               
               self.Xhat.Data = (self.Ex.Data)\Y_temp.Data;
               self.Xhat.stack; 
       end
       
       function self = estimate_Xhat(self)
           
           Y_temp = MEASURE_IMAGESTACK(self.Y.Data);
           Y_temp.Data(find(Y_temp.Data(:)<0))=0;
           
           Y_temp.preprocess;
           Y_temp.columnize;
           
           
           
           try             
               load(sprintf('A_waverec2_%dx%d.mat',[self.Xhat.Ny self.Xhat.Nx]))
           catch
               A_waverec2 =getWaveletmatrices(self.Xhat.Ny,self.Xhat.Nx);
               save(sprintf('A_waverec2_%dx%d.mat',[self.Xhat.Ny self.Xhat.Nx]),'A_waverec2');
           end
           A_wavedec2 = inv(A_waverec2);
           
           x0 = (self.Ex.Data)\Y_temp.Data;
           x0 = A_wavedec2*x0;
           
           self.Xhat.Data = estimate_Xhat_cvx(Y_temp.Data,self.Ex.Data,A_wavedec2);
           %self.Xhat.Data = estimate_Xhat_twist(Y_temp.Data,self.Ex.Data,A_waverec2,x0);           
           self.Xhat.stack; 
           % imagesc(self.Xhat.Data);axis image;colormap hot; set(gca,'xtick',[]);set(gca,'ytick',[])       
       end       
   end
end

           
           
