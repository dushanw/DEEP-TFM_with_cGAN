% 20181019 by Dushan N. Wadduwage
% data class definition read image function

function Data = readImages_cell(dataPath,th)
    k =1;
    display(dataPath)
    while 1
        try 
            I = imread(dataPath,(k-1)*3+1);
            I(find(I<=th))=0;
            Data(:,:,k) = I;
            k = k+1;
        catch
            break
        end
    end
    Data = Data(:,:,1:end-50);% remove last frames to be in the cell region
end
