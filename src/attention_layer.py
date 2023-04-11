from torch.nn import (
    Module,
    Conv2d, 
    ReLU, 
    Sigmoid
)


class AttentionLayer(Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        assert in_channels%2 == 0, "Attention layer: in_channels has to be even"
        self.conv1 = Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.reLU = ReLU()
        self.conv3 = Conv2d(in_channels//2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x, skip):
        x = self.conv1(x)
        skip = self.conv2(skip)
        attention_layer = self.reLU(x + skip)
        attention_layer = self.conv3(attention_layer)
        attention_layer = self.sigmoid(attention_layer)
        return attention_layer