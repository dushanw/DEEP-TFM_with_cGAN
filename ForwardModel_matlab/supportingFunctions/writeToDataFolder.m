
function writeToDataFolder(X0_all)
    mkdir('./_data0_cells/Tr_data/');
    mkdir('./_data0_cells/Tst_data/');
    
    figure
    for i = 1:length(X0_all)-1
        for j=1:size(X0_all{i},3)
            imagesc(X0_all{i}(:,:,j));axis image
            imwrite(X0_all{i}(:,:,j),['./_data0_cells/Tr_data/' sprintf('xo_%d_%d.tif',i,j)]);
            
            drawnow
        end
    end
    
    
    for i = length(X0_all)
        for j=1:size(X0_all{i},3)
            imagesc((X0_all{i}(:,:,j)));axis image
            imwrite(X0_all{i}(:,:,j),['./_data0_cells/Tst_data/' sprintf('xo_%d_%d.tif',i,j)]);
            
            drawnow
        end
    end

    close all
end