#coding=utf-8
from torchvision import models
from pretrainedmodels.models import *
from torchvision.models.inception import InceptionAux
from torch import nn
import types
import sys
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from hubconf import *

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative

def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

def pool_switch(type,x):
    if type=='mac':
        return mac(x)
    elif type=='spoc':
        return spoc(x)
    elif type=='gem':
        return gem(x)
    else:
        return F.adaptive_avg_pool2d(x,-1)
#############  Inception   #############

def get_net_bninception(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes),
            )
    return model

def get_net_bninception_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):                                                                                                    
                                                                                                                                    
    model = bninception(pretrained="imagenet")                                                                                      
    model.global_pool = nn.AdaptiveAvgPool2d(1)                                                                                     
    model.conv1_7x7_s2 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))                          
    model.last_linear = nn.Sequential(                                                                                              
                nn.BatchNorm1d(1024),                                                                                               
                nn.Dropout(0.5),                                                                                                    
                nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),                                                                              
            )                                                                                                                       
    return model 

def get_net_inceptionv3(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = inceptionv3(pretrained="imagenet")
    model.aux_logits = False
    model.Conv2d_1a_3x3.conv = nn.Conv2d(channels, 32, bias=False, kernel_size=3, stride=2)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )
    def logits(self, features):
        x = F.adaptive_avg_pool2d(features,1) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        return x
    model.logits = types.MethodType(logits, model)

    return model
def get_net_inceptionv3_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = inceptionv3(pretrained="imagenet")
    model.aux_logits = False
    model.Conv2d_1a_3x3.conv = nn.Conv2d(channels, 32, bias=False, kernel_size=3, stride=2)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
            )

    def logits(self, features):
        x = F.adaptive_avg_pool2d(features,1) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        return x
    model.logits = types.MethodType(logits, model)

    if run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = F.adaptive_avg_pool2d(x, 1)  # 1 x 1 x 2048
            x = x.view(x.size(0), -1)  # 2048
            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)

    return model

def get_net_inceptionv4(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = inceptionv4(pretrained="imagenet")
    model.features[0].conv = nn.Conv2d(channels, 32,bias=False, kernel_size=3, stride=2)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1536),
                nn.Dropout(0.5),
                nn.Linear(1536, num_classes),
            )
    return model

def get_net_inceptionv4_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = inceptionv4(pretrained="imagenet")
    model.features[0].conv = nn.Conv2d(channels, 32,bias=False, kernel_size=3, stride=2)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1536),
                nn.Dropout(0.5),
                nn.Linear(1536, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    if run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)  # 2048
            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)
    return model

def get_net_inceptionresnetv2(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = inceptionresnetv2(pretrained="imagenet")
    model.conv2d_1a.conv = nn.Conv2d(channels, 32, bias=False, kernel_size=3, stride=2)
    model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.Linear(1536, num_classes),
            )

    return model


def get_net_inceptionresnetv2_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = inceptionresnetv2(pretrained="imagenet")
    model.conv2d_1a.conv = nn.Conv2d(channels, 32, bias=False, kernel_size=3, stride=2)
    model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.Linear(1536, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    return model


def get_net_xception(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = xception(pretrained="imagenet")
    model.conv1 = nn.Conv2d(channels, 32, 3,2, 0, bias=False)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )
    return model
def get_net_xception_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = xception(pretrained="imagenet")
    if channels!=3:
        model.conv1 = nn.Conv2d(channels, 32, 3,2, 0, bias=False)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
            )

    if run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = self.relu(x)

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)

    return model

def get_net_xception_att(model_name, run_type, pool_type, embedding_size, channels, num_classes):

    model = xception(pretrained="imagenet")
    model.conv1 = nn.Conv2d(channels, 32, 3,2, 0, bias=False)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048,  128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )
    model.att_conv1 = nn.Sequential(
        nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    )
    nn.init.xavier_normal(model.att_conv1[0].weight.data)

    model.att_conv2 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, bias=False),
    )
    nn.init.xavier_normal(model.att_conv2[0].weight.data)
    model.score_sigmoid = nn.Sigmoid()


    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        att = self.att_conv1(x)
        att = self.att_conv2(att)
        att = self.score_sigmoid(att)

        x = self.conv3(x)

        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = x+x*att

        x = self.bn4(x)

        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)

    return model

def get_net_alexnet(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = alexnet(pretrained="imagenet")
    model._features[0] = nn.Conv2d(channels, 64, kernel_size=11, stride=4, padding=2)
    model._features[-1] = nn.AdaptiveAvgPool2d(1)

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    del model.linear0
    del model.relu0
    del model.dropout0
    del model.linear1
    del model.relu1
    del model.dropout1

    def features(self, input):
        x = self._features(input)
        return x

    def logits(self, features):
        x = features.view(features.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


#############  ResNet   #############


def _get_basemodel_resnet(model_name, run_type, pool_type, embedding_size):
    if model_name.startswith('resnet18'):
        model = resnet18(pretrained="imagenet")
        last_linear_emb_size = 512
    elif model_name.startswith('resnet34'):
        model = resnet34(pretrained="imagenet")
        last_linear_emb_size = 512
    elif model_name.startswith('resnet50'):
        model = resnet50(pretrained="imagenet")
        last_linear_emb_size = 2048
    elif model_name.startswith('resnet101'):
        model = resnet101(pretrained="imagenet")
        last_linear_emb_size = 2048
    elif model_name.startswith('resnet152'):
        model = resnet152(pretrained="imagenet")
        last_linear_emb_size = 2048
    return model,last_linear_emb_size

def get_net_resnet(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model, last_linear_emb_size = _get_basemodel_resnet(model_name, run_type, pool_type, embedding_size)

    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(last_linear_emb_size),
        nn.Dropout(0.5),
        nn.Linear(last_linear_emb_size, num_classes),
    )

    def logits(self, features):
        if pool_type == 'normal':
            x = self.avgpool(features)
        else:
            x = pool_switch(pool_type, features)
        x_feature = x.view(x.size(0), -1)
        x_class = self.last_linear(x_feature)
        if run_type == 'feature':
            return x_class, x_feature
        else:
            return self.last_linear(x_feature)

    model.logits = types.MethodType(logits, model)

    return model


def get_net_resnet_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model, last_linear_emb_size=_get_basemodel_resnet(model_name, run_type, pool_type, embedding_size)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(last_linear_emb_size),
        nn.Dropout(0.5),
        nn.Linear(last_linear_emb_size, embedding_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(embedding_size, num_classes),
    )

    def logits(self, features):
        if pool_type == 'normal':
            x = self.avgpool(features)
        else:
            x = pool_switch(pool_type, features)
        x_feature = x.view(x.size(0), -1)
        x_class = self.last_linear(x_feature)
        if run_type == 'feature':
            return x_class, x_feature
        else:
            return self.last_linear(x_feature)

    model.logits = types.MethodType(logits, model)

    return model


#############  VGG   #############

def get_net_vgg11_bn(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = vgg11_bn(pretrained="imagenet")
    model._features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
    model._features[-1] = nn.AdaptiveAvgPool2d(1)

    del model.last_linear
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    del model.linear0
    del model.relu0
    del model.dropout0
    del model.linear1
    del model.relu1
    del model.dropout1


    def features(self, input):
        x = self._features(input)
        return x

    def logits(self, features):
        x = features.view(features.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def get_net_vgg16_bn(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = vgg16_bn(pretrained="imagenet")
    model._features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
    model._features[-1] = nn.AdaptiveAvgPool2d(1)

    del model.last_linear
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    del model.linear0
    del model.relu0
    del model.dropout0
    del model.linear1
    del model.relu1
    del model.dropout1


    def features(self, input):
        x = self._features(input)
        return x

    def logits(self, features):
        x = features.view(features.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def get_net_vgg16_bn_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = vgg16_bn(pretrained="imagenet")
    model._features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
    model._features[-1] = nn.AdaptiveAvgPool2d(1)

    del model.last_linear
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )
    del model.linear0
    del model.relu0
    del model.dropout0
    del model.linear1
    del model.relu1
    del model.dropout1


    def features(self, input):
        x = self._features(input)
        return x

    def logits(self, features):
        x = features.view(features.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

#############  ResNext   #############

def get_net_resnext101_32x4d(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = resnext101_32x4d(pretrained="imagenet")
    model.features[0] = nn.Conv2d(channels,64,(7, 7),(2, 2),(3, 3),1,1,bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, num_classes),
    )
    return model
def get_net_resnext101_32x4d_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = resnext101_32x4d(pretrained="imagenet")
    model.features[0] = nn.Conv2d(channels,64,(7, 7),(2, 2),(3, 3),1,1,bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )

    if run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = self.avg_pool(x)  # 1 x 1 x 2048
            x = x.view(x.size(0), -1)  # 2048
            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)

    return model

def get_net_resnext101_64x4d(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = resnext101_64x4d(pretrained="imagenet")
    model.features[0] = nn.Conv2d(channels,64,(7, 7),(2, 2),(3, 3),1,1,bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, num_classes),
    )
    return model

#############  DenseNet   #############

def get_net_densenet121(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = densenet121(pretrained="imagenet")

    model.features.conv0 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),
    )

    def feature_Embedding(self,input):
        x = F.relu(input, inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):

        x = self.features(input)
        x = self.feature_Embedding(x)
        x = self.logits(x)
        return x

    # Modify methods
    model.feature_Embedding = types.MethodType(feature_Embedding, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def get_net_densenet121_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = densenet121(pretrained="imagenet")

    model.features.conv0 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )

    def feature_Embedding(self,input):
        x = F.relu(input, inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):

        x = self.features(input)
        x = self.feature_Embedding(x)
        x = self.logits(x)
        return x

    model.feature_Embedding = types.MethodType(feature_Embedding, model)
    model.logits = types.MethodType(logits, model)
    if run_type == 'feature':

        def forward(self, input):

            x = self.features(input)
            x = self.feature_Embedding(x)
            x = x.view(x.size(0), -1)
            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class, x_feature
        model.forward = types.MethodType(forward, model)
    else:
        def forward(self, input):

            x = self.features(input)
            x = self.feature_Embedding(x)
            x = self.logits(x)
            return x
        model.forward = types.MethodType(forward, model)

    return model

def get_net_densenet161(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = densenet161(pretrained="imagenet")
    model.features.conv0 = nn.Conv2d(channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2208),
        nn.Dropout(0.5),
        nn.Linear(2208, num_classes),
    )

    def feature_Embedding(self,input):
        x = F.relu(input, inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):

        x = self.features(input)
        x = self.feature_Embedding(x)
        x = self.logits(x)
        return x

    # Modify methods
    model.feature_Embedding = types.MethodType(feature_Embedding, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def get_net_densenet161_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = densenet161(pretrained="imagenet")
    model.features.conv0 = nn.Conv2d(channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2208),
        nn.Dropout(0.5),
        nn.Linear(2208, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )

    def feature_Embedding(self,input):
        x = F.relu(input, inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):

        x = self.features(input)
        x = self.feature_Embedding(x)
        x = self.logits(x)
        return x

    # Modify methods
    model.feature_Embedding = types.MethodType(feature_Embedding, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def get_net_densenet169(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = densenet169(pretrained="imagenet")

    model.features.conv0 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1664),
        nn.Dropout(0.5),
        nn.Linear(1664, num_classes),
    )

    def feature_Embedding(self,input):
        x = F.relu(input, inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):

        x = self.features(input)
        x = self.feature_Embedding(x)
        x = self.logits(x)
        return x

    # Modify methods
    model.feature_Embedding = types.MethodType(feature_Embedding, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def get_net_densenet169_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = densenet169(pretrained="imagenet")

    model.features.conv0 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1664),
        nn.Dropout(0.5),
        nn.Linear(1664, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )

    def feature_Embedding(self,input):
        x = F.relu(input, inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):

        x = self.features(input)
        x = self.feature_Embedding(x)
        x = self.logits(x)
        return x

    # Modify methods
    model.feature_Embedding = types.MethodType(feature_Embedding, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

#############  DPN   #############

def get_net_dpn98(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = dpn98(pretrained="imagenet")
    model.features.conv1_1.conv = nn.Conv2d(
        channels, 96, kernel_size=7, stride=2, padding=3, bias=False)
    model.last_linear = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)
    return model
def get_net_dpn107(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = dpn107(pretrained="imagenet+5k")
    model.features.conv1_1.conv = nn.Conv2d(
        channels, 128, kernel_size=7, stride=2, padding=3, bias=False)
    model.last_linear = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)
    return model
def get_net_dpn131(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = dpn131(pretrained="imagenet")
    model.features.conv1_1.conv = nn.Conv2d(
        channels, 128, kernel_size=3, stride=2, padding=1, bias=False)
    model.last_linear = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)

    return model


#############  SeNet   #############

def get_net_senet154(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = senet154(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64, 3, stride=2, padding=1,
                                    bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)

    return model

def get_net_se_resnet50(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnet50(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)
    return model
def get_net_se_resnet50_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnet50(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes))
    return model

def get_net_se_resnet101(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnet101(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)
    return model
def get_net_se_resnet101_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnet101(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes))
    return model

def get_net_se_resnet152(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnet152(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)
    return model

def get_net_se_resnet152_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnet152(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes))
    return model

def get_net_se_resnext50_32x4d(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = se_resnext50_32x4d(pretrained="imagenet")
    model.layer0.conv1 = nn.Conv2d(channels, 64,kernel_size=7, stride=2,
                                    padding=3, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)
    return model


def get_net_pnasnet5large(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = pnasnet5large(pretrained='imagenet',num_classes=1000)
    model.conv_0.conv = nn.Conv2d(channels,  96, kernel_size=3, stride=2, bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(4320, num_classes)

    return model

def get_net_polynet(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    model = polynet(pretrained='imagenet')
    model.stem.conv1[0].conv = nn.Conv2d(channels, 32, 3, stride=2,
                                    bias=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)

    return model

def get_net_wsl(model_name, run_type, pool_type, embedding_size, channels, num_classes):
    if model_name == 'resnext101_32x8d_wsl':
        model = resnext101_32x8d_wsl(True)
    elif model_name == 'resnext101_32x16d_wsl':
        model = resnext101_32x16d_wsl(True)
    elif model_name == 'resnext101_32x32d_wsl':
        model = resnext101_32x32_wsl(True)
    elif model_name == 'resnext101_32x48d_wsl':
        model = resnext101_32x48d_wsl(True)

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, num_classes),
    )

    return model

def get_net(model_name, channels, num_classes):
    run_type = 'train'
    pool_type = 'normal'
    embedding_size = 128

    print('****************** Use {} ******************'.format(model_name, run_type, pool_type, embedding_size))
    if model_name == 'alexnet':
        return get_net_alexnet(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  ResNet  #################
    elif model_name in ['resnet18','resnet34','resnet50','resnet101', 'resnet152']:
        return get_net_resnet(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name in ['resnet34_fc','resnet50_fc','resnet101_fc','resnet152_fc']:
        return get_net_resnet_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)

    ##################  VGG  #################
    elif model_name == 'vgg11_bn':
        return get_net_vgg11_bn(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'vgg16_bn':
        return get_net_vgg16_bn(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'vgg16_bn_fc':
        return get_net_vgg16_bn_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  ResNeXt  #################
    elif model_name == 'resnext101_32x4d':
        return get_net_resnext101_32x4d(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'resnext101_32x4d_fc':
        return get_net_resnext101_32x4d_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'resnext101_64x4d':
        return get_net_resnext101_64x4d(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  DenseNet  #################
    elif model_name == 'densenet121':
        return get_net_densenet121(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'densenet121_fc':
        return get_net_densenet121_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'densenet169':
        return get_net_densenet169(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'densenet169_fc':
        return get_net_densenet169_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'densenet161':
        return get_net_densenet161(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'densenet161_fc':
        return get_net_densenet161_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  Inception  #################
    elif model_name == 'bninception':
        return get_net_bninception(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'bninception_fc':
        return get_net_bninception_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'xception':
        return get_net_xception(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'xception_fc':
        return get_net_xception_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'xception_att':
        return get_net_xception_att(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'inceptionv3':
        return get_net_inceptionv3(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'inceptionv3_fc':
        return get_net_inceptionv3_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'inceptionv4':
        return get_net_inceptionv4(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'inceptionv4_fc':
        return get_net_inceptionv4_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'inceptionresnetv2':
        return get_net_inceptionresnetv2(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  Dual Path Network #################
    elif model_name == 'dpn98':
        return get_net_dpn98(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'dpn107':
        return get_net_dpn107(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'dpn131':
        return get_net_dpn131(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  SeNet #################
    elif model_name == 'senet154':
        return get_net_senet154(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnet50':
        return get_net_se_resnet50(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnet50_fc':
        return get_net_se_resnet50_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnet101':
        return get_net_se_resnet101(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnet101_fc':
        return get_net_se_resnet101_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnet152':
        return get_net_se_resnet152(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnet152_fc':
        return get_net_se_resnet152_fc(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name == 'se_resnext50_32x4d':
        return get_net_se_resnext50_32x4d(model_name, run_type, pool_type, embedding_size, channels, num_classes)#不能用
    ##################  pnasnet5large #################
    elif model_name == 'pnasnet5large':
        return get_net_pnasnet5large(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    ##################  polynet #################
    elif model_name == 'polynet':
        return get_net_polynet(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    elif model_name.find('efficientnet') >= 0:
        return EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name.find('_wsl') >= 0:
        return get_net_wsl(model_name, run_type, pool_type, embedding_size, channels, num_classes)
    else:
        print('Error model {} not found!'.format(model_name, run_type, pool_type, embedding_size))
        sys.exit('0')



if __name__ == '__main__':
    pass
    import sys

    #model = get_net("resnet34", 3, 4)
    model = resnext101_32x8d_wsl(True)
    print(model)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k/1024/1024))
    ##################################i
