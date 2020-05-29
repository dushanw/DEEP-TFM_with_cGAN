% 20181129 by Dushan N. Wadduwage
% data class definition

classdef SIM_EXPERIMENT_ver2 < handle
   properties     
       % simulation parameters
       sl_em
       sl_ex 
       patternType      % E {'hadamardRandom','fullyRandom'};
       patthern_fillFactor
       Nt_mx            % the constructor genereates this many patterns
       N_expTimeFactor  % max counts per pixel (per t)
       dx               % [um] image's pixel size, used to generate sPSF 

       Nx
       Ny
       ny
       nx
       Nt
       
       exPSF
       ExAll
       Ex     
       sPSF
       demagL           % DEMAG object 
       
       A                % Final measurement matrix capturing the system
       
       X0               % Ground truth image (or cube)  
       
       Y_TRAFIX
       Y

       Xhat
       Xhat_TRAFIX
       Xhat_TFM
       Xhat_PSMPM
       Xhat_GT
   end
   methods
       % constructor
       function obj = SIM_EXPERIMENT_ver2(parameterFile)
           % simulation parameters 
           run(parameterFile);                                
           
           obj.sl_em = sl_em;
           obj.sl_ex = sl_ex;
           obj.patternType = patternType; 
           obj.patthern_fillFactor = fillFactor;
           obj.Nt_mx = Nt_mx;
           obj.N_expTimeFactor = N_expTimeFactor; 
           obj.dx = dx;
           obj.Nx = Nx;
           obj.Ny = Ny;
           obj.ny = ny;
           obj.nx = nx;
           
           obj.exPSF = PSF(sim_get_modeled_exPSF([],sl_ex,dx_gt));% no z or ls_ex dependence 
           
           obj.ExAll = CALIB_IMAGESTACK(sim_genExcitationPatterns(Nx,Ny,Nt_mx,patternType,obj.exPSF.psf,dx,dx_gt)); 
           obj.demagL = DEMAG(Ny,Nx,ny,nx);
           
           obj.X0 = IMAGESTACK(zeros(Ny,Nx));
           obj.Xhat = REC_IMAGESTACK(zeros(Ny,Nx));
           obj.Xhat_TRAFIX = REC_IMAGESTACK(zeros(Ny,Nx));
       end
       
       function self = set_X0(self,X0_Data,t_exp)
           self.N_expTimeFactor = t_exp;           
           self.X0.Data = X0_Data*self.N_expTimeFactor; 
       end
       
       function self = build_A(self,Nt,ny,nx,z)
           self.Nt = Nt;
           self.ny = ny;
           self.nx = nx;
           
           self.demagL = DEMAG(self.Ny,self.Nx,self.ny,self.nx);
           self.sPSF = PSF(sim_get_modeled_sPSF(z,self.sl_em,self.dx,self.Nx));
           self.Ex = CALIB_IMAGESTACK(self.ExAll.Data(:,:,1:self.Nt));
           self.Ex.Data = self.Ex.Data/(self.patthern_fillFactor*self.Nt);% this is to conserve the totaol photons
           self.Ex.diagonalize;
           
           Ac = repmat({sparse(self.demagL.A_demag * self.sPSF.A_psf)},self.Nt,1);
           Ac = blkdiag(Ac{:});
           self.A = Ac*self.Ex.Data; 
           
           % self.Y = MEASURE_IMAGESTACK(zeros(self.Ny/self.ny,self.Nx/self.nx,self.Nt));
       end
                  
       function self = image_DEEPTFM(self)
           self.X0.columnize;
           self.Y.Data = self.A * self.X0.Data;            
           self.Y.addPoissonNoise; 
           
           self.X0.stack;
       end
              
       function self = image_TFM(self)
           self.Xhat_TFM = MEASURE_IMAGESTACK(conv2(self.X0.Data,...
                                                    self.sPSF.psf,'same'));
           self.Xhat_TFM.addPoissonNoise;
       end 
       
       function self = image_PSMPM(self)
           self.Xhat_PSMPM = MEASURE_IMAGESTACK(self.X0.Data);
           self.Xhat_PSMPM.addPoissonNoise; 
       end 
       
       function self = image_GT(self)
           self.Xhat_GT = MEASURE_IMAGESTACK(self.X0.Data);           
       end 
       
       function self = estimate_Xhat_wls(self)           
           %A_waverec2 =getWaveletmatrices(self.Xhat.Ny,self.Xhat.Nx);
           load(sprintf('A_waverec2_%dx%d.mat',[self.Ny self.Nx]))          
           A_wavedec2 = inv(A_waverec2);
           self.Xhat.Data = estimate_Xhat_cvx_wls(self.Y.Data,self.A,A_wavedec2);
           
           self.Xhat.stack;            
       end       
       
   end
end

           
           
