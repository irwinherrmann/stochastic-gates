import models.convnet_aig as conv
import models.mobilenet_aig as mobile
import models.unattach_convnet as un_conv
import models.unattach_mobilenet as un_mobile
import models.unattach_mobilenet_filter as mobile_filter
import models.smartergate_convnet as smartergate_conv


def from_modelname(modelname):
    if modelname == 'res':
        model_module = conv
        model = model_module.ResNet50_ImageNet()
        pretrained_end = 'pretrained_imagenet/resnet50.pth'
    elif modelname == 'mobile':
        model_module = mobile
        model = model_module.MobileNetV2()
        pretrained_end = 'pretrained_imagenet/mobilenet_v2.pth.tar'
    elif modelname == 'un_res':
        model_module = un_conv
        model = model_module.ResNet50_ImageNet()
        pretrained_end = 'pretrained_imagenet/resnet50.pth'
    elif modelname == 'un_mobile':
        model_module = un_mobile
        model = model_module.MobileNetV2()
        pretrained_end = 'pretrained_imagenet/mobilenet_v2.pth.tar'
    elif modelname == 'mobile_filter':
        model_module = mobile_filter
        model = model_module.MobileNetV2()
        pretrained_end = 'pretrained_imagenet/mobilenet_v2.pth.tar'
    elif modelname == 'smartergate_res':
        model_module = smartergate_conv
        model = model_module.ResNet50_ImageNet()
        pretrained_end = 'pretrained_imagenet/resnet50.pth'
    else:
        exit(1) # explode

    return model_module, model, pretrained_end
