# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from Transformer import CrossTransformer
import math


class Basic_Residual_Block(nn.Module):
    def __init__(self,BCB_channels=306,BCB_feature=306):
        super( Basic_Residual_Block, self).__init__()
        self.BCB_channels = BCB_channels
        self.input_channel =BCB_feature
        self.conv_layer1 = nn.Sequential( nn.Conv2d(in_channels=self.input_channel, out_channels=self.BCB_channels, kernel_size=3, padding=1), nn.ReLU() )
        self.conv_layer2 = nn.Sequential( nn.Conv2d(in_channels=self.BCB_channels, out_channels=self.BCB_channels, kernel_size=1), nn.ReLU() )
        self.conv_layer3 = nn.Sequential( nn.Conv2d(in_channels=self.BCB_channels, out_channels=self.BCB_channels, kernel_size=3, padding=1), nn.ReLU() )
        self.conv_layer4 = nn.Sequential( nn.Conv2d(in_channels=self.input_channel, out_channels=self.BCB_channels, kernel_size=1), nn.ReLU() )

    def forward(self, BCB_feature):
        feature = self.conv_layer1(BCB_feature)
        feature = self.conv_layer2(feature)
        feature = self.conv_layer3(feature)
        identity_feature = self.conv_layer4(BCB_feature)  
        output = torch.add(feature, identity_feature)
        return output
    

class Multiscale_Block (nn.Module):
    def __init__(self, input_channel=306):
        super(Multiscale_Block, self).__init__()
        self.input_channel = input_channel
        self.conv_sp_atten_1 = nn.Sequential( nn.Conv2d(in_channels=self.input_channel, out_channels=306, kernel_size=3, padding=1), nn.Sigmoid() )
        self.conv_sp_atten_2 = nn.Sequential( nn.Conv2d(in_channels=self.input_channel, out_channels=306, kernel_size=5, padding=2), nn.Sigmoid() )
        self.conv_atten = nn.Sequential( nn.Conv2d(in_channels=2 * self.input_channel, out_channels=306, kernel_size=3, padding=1), nn.Sigmoid() )

        
    def forward(self, Input_features):
        follow_feature1 = self.conv_sp_atten_1(Input_features) 
        follow_feature2 = self.conv_sp_atten_2(Input_features)
        multiscale_feature = torch.cat((follow_feature1, follow_feature2), dim=1)
 	multiscale_feature = self.conv_atten(multiscale_feature)
        return multiscale_feature
    



class CSCANet(nn.Module):
    def __init__(self, input_image_channel=305):
        super(CANet, self).__init__()
        self.input_image_channel = input_image_channel
        self.first_branch_channel = 306//3
        self.input_transformer = math.floor(self.first_branch_channel/6)*6
        self.resinput_transformer = self.first_branch_channel-self.input_transformer

        self.attention_1 = CrossTransformer(self.input_transformer)
        self.attention_2 = CrossTransformer(self.input_transformer)
        self.attention_3 = CrossTransformer(self.input_transformer)
        self.attention_4 = CrossTransformer(self.input_transformer)
        self.attention_5 = CrossTransformer(self.input_transformer)
        self.attention_6 = CrossTransformer(self.input_transformer)
        self.attention_7 = CrossTransformer(self.input_transformer)
        self.attention_8 = CrossTransformer(self.input_transformer)
        self.attention_9 = CrossTransformer(self.input_transformer)
        self.attention_10 = CrossTransformer(self.input_transformer)
        
        
        self.Basic_Residual_1 = Basic_Residual_Block(BCB_feature=306)
        self.Multiscale_Block_1 = Multiscale_Block()
        self.Basic_Residual_2 = Basic_Residual_Block(BCB_feature=306)
       
        
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU() )
        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU() )
        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=102, out_channels=102, kernel_size=1), nn.ReLU() )
        self.conv_layer4 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU() )
        self.conv_layer5 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU() )
        self.conv_layer6 = nn.Sequential(nn.Conv2d(in_channels=102, out_channels=102, kernel_size=1), nn.ReLU() )
                
    def forward(self, Input_HSI):
         # _, band, _, _ = Input_HSI.shape     # 获得HSI的通道数

         first_branch_channel = self.first_branch_channel
         input_transformer = self.input_transformer

         first_branch_input = Input_HSI[:, 0:first_branch_channel, :, :]
         second_branch_input = Input_HSI[:, first_branch_channel:2*first_branch_channel, :, :]
         third_branch_input = second_branch_input
         third_branch_input[:, 0:first_branch_channel - 1, :, :] = Input_HSI[:, 2*first_branch_channel:, :, :]
         third_branch_input[:, first_branch_channel - 1, :, :] = third_branch_input[:, first_branch_channel - 2, :, :]         
	

         first_branch_output = torch.cat((self.attention_1(first_branch_input, first_branch_input), self.attention_2(first_branch_input, second_branch_input),), dim=1)
         first_branch_output = self.conv_layer1(first_branch_output) + first_branch_input

         second_branch_output = torch.cat((self.attention_3(second_branch_input, second_branch_input), self.attention_4(second_branch_input, third_branch_input)), dim=1)
         second_branch_output = self.conv_layer2(second_branch_output) + second_branch_input

         third_branch_output = self.attention_5(third_branch_input, third_branch_input) + third_branch_input
	 third_branch_output = self.conv_layer3(third_branch_output)


         first_branch_output1 = torch.cat((self.attention_6(first_branch_output, first_branch_output), self.attention_7(first_branch_output, second_branch_output),), dim=1)
         first_branch_output = self.conv_layer4(first_branch_output1) + first_branch_output

         second_branch_output1 = torch.cat((self.attention_8(second_branch_output, second_branch_output), self.attention_9(second_branch_output, third_branch_output)), dim=1)
         second_branch_output = self.conv_layer5(second_branch_output1) + second_branch_output

         third_branch_output = self.attention_10(third_branch_output, third_branch_output) + third_branch_output
	 third_branch_output = self.conv_layer6(third_branch_output)

        
         merge_output = torch.cat((first_branch_output, second_branch_output, third_branch_output), dim=1)
         merge_1 = self.Basic_Residual_1(merge_output)
         merge_1 = self.Multiscale_Block_1(merge_1)
         merge_1 = self.Basic_Residual_2(merge_1)

         output = merge_1[:, 0:305, :, :]
         return output
     
        
if __name__ == '__main__':

    mcplb = CSCANet(input_image_channel=305).cuda()
   

 











