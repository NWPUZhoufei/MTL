import torch 
from torch import nn

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv = nn.ModuleList([
            # block1
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1),  stride=(2, 1, 1)),
            # block2
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            # block3
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            # block4
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
            ])
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        batch_size, channel, dims, weight, height = inputs.size()
        for i, block in enumerate(self.conv):
            inputs = block(inputs)
            
        out = inputs.reshape(batch_size, -1)
        return out


if __name__ == '__main__':

    model = Embedding()
    data = torch.rand(80, 100, 9, 9)

    outs = model(data)
    print(outs.shape)

    

    




