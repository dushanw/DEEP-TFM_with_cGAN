% 20181019 by Dushan N. Wadduwage
% data class definition read image function

function [Data, O_wf] = readImages(dataPath,dataNameStem,crop_rect,Nx,Ny,Nt,EmGain)
       
    % background pixel indices
    Temp = imread([dataPath dataNameStem '_part1.tif'],1);
    Temp(:)=0;
    Temp(crop_rect(2)-10:crop_rect(2)+crop_rect(4)+10,crop_rect(1)-10:crop_rect(1)+crop_rect(3)+10)=1;
    pixelInds_outside = find(Temp(:)==0);

    Data = double(zeros(Nx,Ny,Nt));
    Data_bg = [];
    for i=1:Nt/2
        i
        Data_temp = imread([dataPath dataNameStem '_part1.tif'],i);
        Data_bg = [Data_bg; Data_temp(pixelInds_outside)];
        Data(:,:,i) = imresize(imcrop(Data_temp,crop_rect),[Nx Ny]);
        
        Data_temp = imread([dataPath dataNameStem '_part2.tif'],i);
        Data_bg = [Data_bg; Data_temp(pixelInds_outside)];
        Data(:,:,Nt/2+i) = imresize(imcrop(Data_temp,crop_rect),[Nx Ny]);        
    end

    O_wf =  imresize(imcrop(imread([dataPath dataNameStem '_part1.tif'],129),crop_rect),[Nx Ny]);
    Data_bg = mean(Data_bg);
    
    Data = (Data - Data_bg)/EmGain;

end
