import torch.nn as nn
import torch.nn.functional as F


class CBR2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=True):
        super(CBR2d, self).__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),  # 64 x 512 x 512
            nn.BatchNorm2d(out_channel),
            nn.SELU(bias),
        )

    def forward(self, x):
        x = self.cbr(x)
        return x


class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv, self).__init__()

        self.encoder = nn.Sequential(
            CBR2d(1, 64, 3, 1, 1),      # 64 512 512
            nn.MaxPool2d(2, 2),         # 64 256 256
            CBR2d(64, 128, 3, 1, 1),    # 128 256 256
            nn.MaxPool2d(2, 2),         # 128 128 128
            CBR2d(128, 256, 3, 1, 1),   # 256 128 128
            nn.MaxPool2d(2, 2),         # 256 64 64
            CBR2d(256, 512, 3, 1, 1),   # 512 64 64
            nn.MaxPool2d(2, 2),         # 512 32 32
            CBR2d(512, 512, 3, 1, 1),   # 512 32 32
            nn.MaxPool2d(2, 2),         # 512 16 16
            CBR2d(512, 512, 3, 1, 1),   # 512 16 16
            CBR2d(512, 512, 3, 1, 1),   # 512 16 16
            nn.MaxPool2d(2, 2),         # 512 8 8
        )

        self.decoder = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.SELU(True),
            nn.Linear(1024, 500),
            nn.SELU(True),
            nn.Linear(500, 1024),
            nn.SELU(True),
            nn.Linear(1024, 512 * 512),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.view([x.size(0), -1])
        out = self.decoder(out)
        out = out.view([x.size(0), 1, 512, 512])
        return out


class AutoEncoderConv3(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(11 * 11 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 320)
        self.dec_linear_2 = nn.Linear(320, 3136)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))

        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, 56, 56])
        return out
