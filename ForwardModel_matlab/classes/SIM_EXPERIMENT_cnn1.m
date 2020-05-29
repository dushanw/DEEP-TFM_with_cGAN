% 20181129 by Dushan N. Wadduwage
% 20191230 updated by Dushan N. Wadduwage (original SIM_EXPERIMENT_ver2)
% data class definition

classdef SIM_EXPERIMENT_cnn1 < handle
   properties     
       % image parameters
       sl_em
       sl_ex 
       Nt_mx            % the constructor genereates this many patterns
       dx               % [um] image's pixel size, used to generate sPSF 

       Nx
       Ny
       ny
       nx
       Nt
       z
       
       exPSF
       ExAll
       Ex     
       sPSF
       demagL           % DEMAG object 
       
       X0
       X0_bg
       X0_sig
       X0_gain
       
       A                % Final measurement matrix capturing the system
              
       YAll
       Y

       Xhat
       Xhat_TFM       
   end
   methods
       % constructor
       function obj = SIM_EXPERIMENT_cnn1(parameterFile)
           % simulation parameters 
           run(parameterFile);                                
           
           obj.sl_em    = sl_em;
           obj.sl_ex    = sl_ex;                      
           obj.Nt_mx    = Nt_mx;           
           obj.dx       = dx;
           obj.Nx       = Nx;
           obj.Ny       = Ny;
           obj.ny       = ny;
           obj.nx       = nx;
           obj.z        = z;
           
           obj.exPSF    = PSF(sim_get_modeled_exPSF([],sl_ex,dx));% no z or ls_ex dependence 
           
           obj.ExAll    = CALIB_IMAGESTACK([],extPath,extNameStem,Nx,Ny,Nt_mx,EmGain,crop_rect_name_ex); 
           obj.YAll     = MEASURE_IMAGESTACK([],dataPath,dataNameStem,Nx/nx,Ny/ny,Nt_mx,EmGain,crop_rect_name);
           
           load YpreProcess;
           obj.X0_bg    = Data_bg;
           obj.X0_sig   = Data_avgSig;
           obj.X0_gain  = EmGain;
           
           obj.demagL   = DEMAG(Ny,Nx,ny,nx);
           
           obj.X0       = IMAGESTACK(zeros(Ny,Nx)); 
           obj.Xhat     = REC_IMAGESTACK(zeros(Ny,Nx));           
       end
       
       function self = set_X0(self,maxCounts) 
           self.X0_sig = maxCounts;
           self.X0.Data = gen_beadsIn2D(self.Nx,self.Ny,0)*self.X0_sig; 
       end
       
       function self = build_A(self,Nt,ny,nx,z)
           self.Nt = Nt;
           self.ny = ny;
           self.nx = nx;
           self.z = z;
           
           self.demagL = DEMAG(self.Ny,self.Nx,self.ny,self.nx);
           self.sPSF = PSF(sim_get_modeled_sPSF(z,self.sl_em,self.dx,self.Nx));
           self.Ex = CALIB_IMAGESTACK(self.ExAll.Data(:,:,1:self.Nt));          
           self.Ex.diagonalize;
           
           Ac = repmat({sparse(self.demagL.A_demag * self.sPSF.A_psf)},self.Nt,1);
           Ac = blkdiag(Ac{:});
           self.A = Ac*self.Ex.Data; 
           
           self.Y = MEASURE_IMAGESTACK(zeros(self.Ny/self.ny,self.Nx/self.nx,self.Nt));
       end       
       
       function self = build_A_efficient(self,Nt,ny,nx,z)
           self.Nt = Nt;
           self.ny = ny;
           self.nx = nx;
           self.z = z;
           
           self.demagL = DEMAG(self.Ny,self.Nx,self.ny,self.nx);
           self.sPSF = PSF(sim_get_modeled_sPSF(z,self.sl_em,self.dx,self.Nx));
           self.Ex = CALIB_IMAGESTACK(self.ExAll.Data(:,:,1:self.Nt)); 
           self.Ex.Data = self.Ex.Data/max(self.Ex.Data(:));
           % self.Ex.preprocess;
           % self.Ex.diagonalize;
           
           Ac = self.demagL.A_demag * self.sPSF.A_psf;           
 
           for i=1:Nt
               i
               Ex_now = IMAGESTACK(self.Ex.Data(:,:,i));
               Ex_now.diagonalize; 
               A{i} = Ac*Ex_now.Data; 
           end
           self.A = cat(1,A{:});
           self.Y = MEASURE_IMAGESTACK(zeros(self.Ny/self.ny,self.Nx/self.nx,self.Nt));
       end
       
       function self = image_DEEPTFM(self)
           self.X0.columnize;
           self.Y.Data = self.A * self.X0.Data;            
           self.Y.addPoissonNoise; 
           
           self.Y.stack;
           self.X0.stack;
       end
       
       function self = save_DEEPTFM(self,saveToFolder,nameStem)
           
           Y_temp = self.Y.Data;
           Y_temp = Y_temp*self.X0_gain+self.X0_bg;
           Y_temp = uint16(Y_temp);
           for i=1:self.Nt
              imwrite(Y_temp(:,:,i),[saveToFolder nameStem sprintf('_%d.png',i)]); 
           end
           
           X0_temp = uint16(self.X0.Data);
           imwrite(X0_temp,[saveToFolder nameStem sprintf('_gt.png')]);
           
       end
       
       function self = save_Yreal(self,saveToFolder,nameStem)
           
           Y_temp = self.YAll.Data(:,:,1:self.Nt);
           Y_temp(find(Y_temp(:)<0))=0;
           
           Y_temp = Y_temp*self.X0_gain+self.X0_bg;
           Y_temp = uint16(Y_temp);
           for i=1:self.Nt
              imwrite(Y_temp(:,:,i),[saveToFolder nameStem sprintf('_%d.png',i)]); 
           end
                    
       end
       
       function self = estimate_Xhat_wls(self)           
           %A_waverec2 =getWaveletmatrices(self.Xhat.Ny,self.Xhat.Nx);
           load(sprintf('A_waverec2_%dx%d.mat',[self.Ny self.Nx]))          
           A_wavedec2 = inv(A_waverec2);
           Y_temp = MEASURE_IMAGESTACK(self.Y.Data(:,:,1:self.Nt));
           % Y_temp.preprocess;           
           Y_temp.columnize;
           self.Xhat.Data = estimate_Xhat_cvx_wls(Y_temp.Data,self.A,A_wavedec2);
           
           self.Xhat.stack;            
       end
       
       function self = estimate_Xhat0(self)
               Y_temp = MEASURE_IMAGESTACK(self.Y.Data(:,:,1:self.Nt));           
               Y_temp.Data(find(Y_temp.Data(:)<0))=0;
                      
               %Y_temp.preprocess;           
               Y_temp.columnize;
               
               A = full(self.A);
               A1 = A'*A;
               Y1 = A'*Y_temp.Data;
               
               self.Xhat.Data = A1\Y1;
               self.Xhat.stack; 
       end
       
       function self = estimate_Xhat_cvx(self)
               Y_temp = MEASURE_IMAGESTACK(self.Y.Data(:,:,1:self.Nt));           
               Y_temp.Data(find(Y_temp.Data(:)<0))=0;
                      
               %Y_temp.preprocess;           
               Y_temp.columnize;
               
               A = full(self.A);
               A1 = A'*A;
               Y1 = A'*Y_temp.Data;
               
               self.Xhat.Data = estimate_Xhat_cvx(Y1,A1,[]);
               self.Xhat.stack; 
       end
       
       
   end
end

           
           
