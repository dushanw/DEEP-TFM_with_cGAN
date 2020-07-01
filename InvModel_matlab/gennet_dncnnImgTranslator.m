function lgraph = gennet_dncnnImgTranslator(size_In)

    net0_dncnn = denoisingNetwork('dncnn');
    lgraph = layerGraph(net0_dncnn.Layers);
    
    lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer(size_In,'Name','InputLayer','Normalization','none'));        
    lgraph = replaceLayer(lgraph,'Conv1',convolution2dLayer([3 3],64,'Name','Conv1','NumChannels',size_In(3),'Padding',[1 1 1 1],'stride',[1  1]));        
    
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_2_5'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_5_10'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_10_15'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_15_19'));
    
    lgraph = disconnectLayers(lgraph,'BNorm5','ReLU5');
    lgraph = disconnectLayers(lgraph,'BNorm10','ReLU10');
    lgraph = disconnectLayers(lgraph,'BNorm15','ReLU15');
    lgraph = disconnectLayers(lgraph,'BNorm19','ReLU19');
    
    lgraph = connectLayers(lgraph,'BNorm5','add_2_5/in1');
    lgraph = connectLayers(lgraph,'ReLU2','add_2_5/in2');
    lgraph = connectLayers(lgraph,'add_2_5','ReLU5');
    
    lgraph = connectLayers(lgraph,'BNorm10','add_5_10/in1');
    lgraph = connectLayers(lgraph,'ReLU5','add_5_10/in2');
    lgraph = connectLayers(lgraph,'add_5_10','ReLU10');
    
    lgraph = connectLayers(lgraph,'BNorm15','add_10_15/in1');
    lgraph = connectLayers(lgraph,'ReLU10','add_10_15/in2');
    lgraph = connectLayers(lgraph,'add_10_15','ReLU15');
    
    lgraph = connectLayers(lgraph,'BNorm19','add_15_19/in1');
    lgraph = connectLayers(lgraph,'ReLU15','add_15_19/in2');
    lgraph = connectLayers(lgraph,'add_15_19','ReLU19');        
end





