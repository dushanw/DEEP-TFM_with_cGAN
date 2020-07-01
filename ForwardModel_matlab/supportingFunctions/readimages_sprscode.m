% 20181019 by Dushan N. Wadduwage
% data class definition read image function

function [Data, O_wf] = readimages_sprscode(dataPath,dataNameStem,crop_rect,Nx,Ny,Nt,EmGain)

    % crop_rect{1} = [206 296 64 64];

    Data = double(zeros(Nx,Ny,Nt));
    Data_bg = [];
    for i=1:Nt/2
        i
        Data_temp = imread([dataPath dataNameStem '_1.tif'],i);
        Data_bg_temp = imcrop(Data_temp,crop_rect{2});
        Data_bg = [Data_bg; Data_bg_temp(:)];
        Data(:,:,i) = imresize(imcrop(Data_temp,crop_rect{1}),[Nx Ny]);
        
        Data_temp = imread([dataPath dataNameStem '_2.tif'],i);
        Data_bg_temp = imcrop(Data_temp,crop_rect{2});
        Data_bg = [Data_bg; Data_bg_temp(:)];
        Data(:,:,Nt/2+i) = imresize(imcrop(Data_temp,crop_rect{1}),[Nx Ny]);        
    end
    save('Yreal0.mat','Data');
    O_wf =  imresize(imcrop(imread([dataPath dataNameStem '_1.tif'],1),crop_rect{1}),[Nx Ny]);
    Data_bg = mean(Data_bg);
%    Data_bg = 300;
    
    Data = (Data - Data_bg)/EmGain;
    
    Data_avgSig = mean(Data(:));
    save('YpreProcess.mat','Data_bg','Data_avgSig','EmGain');
end
