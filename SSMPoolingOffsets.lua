local SSMPoolingOffsets, parent = torch.class('jzt.SSMPoolingOffsets', 'nn.Module')

function SSMPoolingOffsets:__init(kW, kH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.gridX = torch.linspace(-1,1,kW)
   self.gridY = torch.linspace(-1,1,kH) 

end

function SSMPoolingOffsets:updateOutput(input)
    if input:type() == 'torch.CudaTensor' then 
        self.softmax = torch.CudaTensor()  
        self.gridX = self.gridX:cuda() 
        self.gridY = self.gridY:cuda() 
        jzt.SSMPoolingOffsets_updateOutput(self, input)
    else
        local inputSize = input:size() 
        local nOutputPlane = 2*inputSize[2] 
        local nOutputCols = math.floor(inputSize[4]/kW)
        local nOutputRows = math.floor(inputSize[3]/kH) 
        local maxW = nOutputCols*kW
        local maxH = nOutputRows*kH 
        
        self.output:resize(inputSize[1],nOutputPlane,nOutputRows,nOutputCols):typeAs(input)
        self.softmax = self.softmax or torch.Tensor():resize(inputSize[1],inputSize[2],maxH,maxW):typeAs(input) 

        for batch = 1,inputSize[1] do 
            for inplane = 1,inputSize[2] do 
                local oi = 1 
                for i = 1,maxH,kH do 
                    local oj = 1 
                    for j = 1,maxW,kW do 
                       
                        local pool_sum = 0 
                        local dx = 0
                        local dy = 0 
        
                        for h = 0,kH-1 do 
                            for w = 0,kW-1 do
                                pool_sum = pool_sum + math.exp(input[batch][inplane][i+h][j+w])                      
                            end
                        end
                       
                        for h = 0,kH-1 do 
                            for w = 0,kW-1 do 
                                local val = math.exp(input[batch][inplane][i+h][j+w])/pool_sum                      
                                self.softmax[batch][inplane][i+h][j+w] = val                    
                                dx = dx + self.gridX[w+1]*val 
                                dy = dy + self.gridY[h+1]*val
                            end
                        end
                       
                        self.output[batch][2*inplane-1][oi][oj] = dx 
                        self.output[batch][2*inplane][oi][oj] = dy 
                        
                        oj = oj + 1 
                    end
                    oi = oi + 1 
                end
            end
        end
    end
   return self.output
end

function SSMPoolingOffsets:updateGradInput(input, gradOutput)
    if input:type() == 'torch.CudaTensor' then 
        jzt.SSMPoolingOffsets_updateGradInput(self, input, gradOutput)
    else   
        self.gradInput = torch.Tensor():resizeAs(input):fill(0)  
        local inputSize = input:size() 
        local nOutputPlane = 2*inputSize[2] 
        local nOutputCols = math.floor(inputSize[4]/kW)
        local nOutputRows = math.floor(inputSize[3]/kH) 
        local maxW = nOutputCols*kW
        local maxH = nOutputRows*kH 
        
        for batch = 1,inputSize[1] do 
            for inplane = 1,inputSize[2] do 
                local oi = 1 
                for i = 1,maxH,kH do 
                    local oj = 1 
                    for j = 1,maxW,kW do 
                       
                        local pool_sum_X = 0  
                        local pool_sum_Y = 0  
                        
                        for h = 0,kH-1 do 
                            for w = 0,kW-1 do
                                pool_sum_X = pool_sum_X + self.softmax[batch][inplane][i+h][j+w]*self.gridX[w+1] 
                                pool_sum_Y = pool_sum_Y + self.softmax[batch][inplane][i+h][j+w]*self.gridY[h+1] 
                            end
                        end
        
                        local gradOutput_X = gradOutput[batch][2*inplane-1][oi][oj]  
                        local gradOutput_Y = gradOutput[batch][2*inplane][oi][oj]  
                        
                        for h = 0,kH-1 do 
                            for w = 0,kW-1 do 
                                local softmax_pool = self.softmax[batch][inplane][i+h][j+w] 
                                local gradInput = softmax_pool * (gradOutput_X * (self.gridX[w+1] - pool_sum_X) + 
                                                                  gradOutput_Y * (self.gridY[h+1] - pool_sum_Y))   
                                self.gradInput[batch][inplane][i+h][j+w] = gradInput 
                            end
                        end
                       
                        oj = oj + 1 
                    end
                    oi = oi + 1 
                end
            end
        end
            

    end

   return self.gradInput
end

function SSMPoolingOffsets:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end
