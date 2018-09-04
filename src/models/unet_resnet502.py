# from dask.tests.test_base import np
# from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
# from keras.models import Sequential
#
# from common import *
# #  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# #  resnet18 :  BasicBlock, [2, 2, 2, 2]
# #  resnet34 :  BasicBlock, [3, 4, 6, 3]
# #  resnet50 :  Bottleneck  [3, 4, 6, 3]
# #
#
# # https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
# # https://github.com/ternaus/TernausNetV2
# # https://github.com/neptune-ml/open-solution-salt-detection
# # https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution
#
#
# ##############################################################3
# #  https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py
# #  https://pytorch.org/docs/stable/torchvision/models.html
#
# import torchvision
#
# # ConvBn2D
# def conv2d_bn(filters, kernel_size, strides, padding, use_bias=False):
#     def __apply(inp):
#         x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(inp)
#         x = BatchNormalization()(x)
#         return x
#     return __apply
#
# def SEScale(reduction, channel):
#     def __apply(inp):
#         x = GlobalAveragePooling2D()(inp)
#         # TODO Reshape?
#         x = Conv2D(filters=reduction, kernel_size=1, padding=0)(x)
#         x = Activation('relu')(x)
#         x = Conv2D(filters=channel, kernel_size=1, padding=0)(x)
#         x = Activation('sigmoid')(x)
#         return x
#     return __apply
#
#
# def SEBottleneck(planes, out_planes, reduction, is_downsample=False, stride=1):
#     def __apply(inp):
#         x = conv2d_bn(planes, kernel_size=1, padding=0, stride=1)(inp)
#         x = Activation('relu')(x)
#         x = conv2d_bn(planes, kernel_size=3, padding=1, stride=stride)(x)
#         x = Activation('relu')(x)
#         x = conv2d_bn(out_planes, kernel_size=1, padding=0, stride=1)(x)
#         x = x * SEScale(out_planes, reduction)(x)
#         if is_downsample:
#             x += conv2d_bn(out_planes, kernel_size=1, padding=0, stride=stride)(x)
#         else:
#             x += x
#         x = Activation('relu')(x)
#         return x
#
#     return __apply
#
#
# # layers ##---------------------------------------------------------------------
#
# def make_layer(in_planes, planes, out_planes, reduction, num_blocks, stride):
#     layers = []
#     layers.append(SEBottleneck(in_planes, planes, out_planes, reduction, is_downsample=True, stride=stride))
#     for i in range(1, num_blocks):
#         layers.append(SEBottleneck(out_planes, planes, out_planes, reduction))
#     model = Sequential(layers=layers)
#     return model
#
#
# def Decoder(in_channels, channels, out_channels):
#     def __apply(inp):
#         x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)  # False
#         x = conv2d_bn(channels, kernel_size=3, padding=1)(inp)
#         x = Activation('relu')(x)
#         x = conv2d_bn(out_channels, kernel_size=3, padding=1)
#         x = Activation('relu')(x)
#         return x
#     return __apply
#
#
# ###########################################################################################3
#
# class UNetSEResNet50(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Sequential(
#             ConvBn2d(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(inplace=True),
#         )
#
#         self.encoder2 = make_layer(64, 64, 256, reduction=16, num_blocks=3, stride=1)  # out =  64*4 =  256
#         self.encoder3 = make_layer(256, 128, 512, reduction=32, num_blocks=4, stride=2)  # out = 128*4 =  512
#         self.encoder4 = make_layer(512, 256, 1024, reduction=64, num_blocks=6, stride=2)  # out = 256*4 = 1024
#         self.encoder5 = make_layer(1024, 512, 2048, reduction=128, num_blocks=3, stride=2)  # out = 512*4 = 2048
#
#         self.center = nn.Sequential(
#             ConvBn2d(2048, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             ConvBn2d(512, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         self.decoder5 = Decoder(2048 + 256, 512, 256)
#         self.decoder4 = Decoder(1024 + 256, 512, 256)
#         self.decoder3 = Decoder(512 + 256, 256, 64)
#         self.decoder2 = Decoder(256 + 64, 128, 128)
#         self.decoder1 = Decoder(128, 128, 32)
#
#         self.logit = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=1, padding=0),
#         )
#
#     def forward(self, x):
#         # batch_size,C,H,W = x.shape
#
#         mean = [0.40784314, 0.45882353, 0.48235294]
#         x = torch.cat([
#             (x - mean[0]) * 255,
#             (x - mean[1]) * 255,
#             (x - mean[2]) * 255,
#         ], 1)
#
#         x = self.conv1(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#
#         e2 = self.encoder2(x)  # ; print('e2',e2.size())
#         e3 = self.encoder3(e2)  # ; print('e3',e3.size())
#         e4 = self.encoder4(e3)  # ; print('e4',e4.size())
#         e5 = self.encoder5(e4)  # ; print('e5',e5.size())
#
#         # f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
#         # f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)#False
#         # f = self.center(f)                       #; print('center',f.size())
#         f = self.center(e5)
#
#         f = self.decoder5(torch.cat([f, e5], 1))  # ; print('d5',f.size())
#         f = self.decoder4(torch.cat([f, e4], 1))  # ; print('d4',f.size())
#         f = self.decoder3(torch.cat([f, e3], 1))  # ; print('d3',f.size())
#         f = self.decoder2(torch.cat([f, e2], 1))  # ; print('d2',f.size())
#         f = self.decoder1(f)  # ; print('d1',f.size())
#
#         # f = F.dropout2d(f, p=0.20)
#         logit = self.logit(f)  # ; print('logit',logit.size())
#         return logit
#
#     ##-----------------------------------------------------------------
#
#     def criterion(self, logit, truth):
#
#         # loss = PseudoBCELoss2d()(logit, truth)
#         loss = FocalLoss2d()(logit, truth, type='sigmoid')
#         # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
#         return loss
#
#     # def criterion(self,logit, truth):
#     #
#     #     loss = F.binary_cross_entropy_with_logits(logit, truth)
#     #     return loss
#
#     def metric(self, logit, truth, threshold=0.5):
#         prob = F.sigmoid(logit)
#         dice = accuracy(prob, truth, threshold=threshold, is_average=True)
#         return dice
#
#     def set_mode(self, mode):
#         self.mode = mode
#         if mode in ['eval', 'valid', 'test']:
#             self.eval()
#         elif mode in ['train']:
#             self.train()
#         else:
#             raise NotImplementedError
#
#
# SaltNet = UNetSEResNet50
#
#
# ### run ##############################################################################
#
#
# def run_check_net():
#     batch_size = 8
#     C, H, W = 1, 128, 128
#
#     input = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
#     truth = np.random.choice(2, (batch_size, C, H, W)).astype(np.float32)
#
#     # ------------
#     input = torch.from_numpy(input).float().cuda()
#     truth = torch.from_numpy(truth).float().cuda()
#
#     # ---
#     net = SaltNet().cuda()
#     net.set_mode('train')
#     # print(net)
#     # exit(0)
#
#     # net.load_pretrain('/root/share/project/kaggle/tgs/data/model/resnet50-19c8e357.pth')
#
#     logit = net(input)
#     loss = net.criterion(logit, truth)
#     dice = net.metric(logit, truth)
#
#     print('loss : %0.8f' % loss.item())
#     print('dice : %0.8f' % dice.item())
#     print('')
#
#     # dummy sgd to see if it can converge ...
#     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
#                           lr=0.1, momentum=0.9, weight_decay=0.0001)
#
#     # optimizer = optim.Adam(net.parameters(), lr=0.001)
#
#     i = 0
#     optimizer.zero_grad()
#     while i <= 500:
#
#         logit = net(input)
#         loss = net.criterion(logit, truth)
#         dice = net.metric(logit, truth)
#
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         if i % 20 == 0:
#             print('[%05d] loss, dice  :  %0.5f,%0.5f' % (i, loss.item(), dice.item()))
#         i = i + 1
#
#
# ########################################################################################
# if __name__ == '__main__':
#     print('%s: calling main function ... ' % os.path.basename(__file__))
#
#     run_check_net()
#
#     print('sucessful!')