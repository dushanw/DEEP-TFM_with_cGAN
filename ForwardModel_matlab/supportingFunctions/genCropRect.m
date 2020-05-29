

function crop_rect = genCropRect(PatternPath,patternNameStem,crop_rect_name)
    
    try I_temp = imread([PatternPath patternNameStem],1);% for single file
    catch
        try I_temp = imread([PatternPath patternNameStem '_part1.tif'],129);
        catch I_temp = imread([PatternPath patternNameStem '_1.tif'],1);
        end
    end
    
        
    [I_cropped,crop_rect_fg] = imcrop(I_temp*50);% crop ROI manaually
    [I_cropped,crop_rect_bg] = imcrop(I_temp*50);% crop ROI manaually
    crop_rect{1}=crop_rect_fg;
    crop_rect{2}=crop_rect_bg;
    close all
    save(crop_rect_name,'crop_rect');
    
end