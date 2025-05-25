import torch 
import torch.nn as nn

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # three branch
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.identity = nn.Identity() if in_channels == out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv3x3(x)
        out += self.conv1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return self.relu(out)

 
    def reparameterize(self):
        # 3x3、1x1 and Identity parameter
        kernel_3x3, bias_3x3 = self._get_merged_kernel_bias()
        merged_conv = nn.Conv2d(
            in_channels=self.conv3x3.in_channels,
            out_channels=self.conv3x3.out_channels,
            kernel_size=3,
            padding=1
        )
        merged_conv.weight.data = kernel_3x3
        merged_conv.bias.data = bias_3x3
        return merged_conv,self.conv3x3.weight, self.conv3x3.bias
