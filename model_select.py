import models.convnet as conv
import models.unattach_convnet as un_conv

import models.test_models.test_convnet as testres
import models.test_models.test_unattach_convnet as testunres


def get_cifar(modelname):
    if modelname == 'res':
        model_module = conv
        model = model_module.ResNet110_cifar()
    elif modelname == 'un_res':
        model_module = un_conv
        model = model_module.ResNet110_cifar()
    else:
        exit(1) # explode
    return model_module, model

def get_imgnet(modelname):
    if modelname == 'res':
        model_module = conv
        model = model_module.ResNet50_ImageNet()
        pretrained_end = 'pretrained_imagenet/resnet50.pth'
    elif modelname == 'un_res':
        model_module = un_conv
        model = model_module.ResNet50_ImageNet()
        pretrained_end = 'pretrained_imagenet/resnet50.pth'
    else:
        exit(1) # explode
    return model_module, model, pretrained_end

def get_test(modelname):
    is_independent = 'un' in modelname
    is_res101 = 'res101' in modelname

    if is_independent:
        model_module = testunres
    else:
        model_module = testres

    if is_res101:
        model = model_module.ResNet101_ImageNet()
    else:
        model = model_module.ResNet50_ImageNet()

    return model_module, model

