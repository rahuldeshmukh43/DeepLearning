from yacs.config import CfgNode as CN

_CN = CN()

#Network Config
_CN.NETWORK = CN()

#BACKBONE
#resnet config
_CN.NETWORK.BACKBONE='resnet101'
_CN.NETWORK.PRETRAINED=True
#_CN.NETWORK.FIX_WTS=True

#ConvNet4D config
_CN.NETWORK.CONVNET4D = CN()
#smaller network
_CN.NETWORK.CONVNET4D.KERNEL_SIZE=3
_CN.NETWORK.CONVNET4D.OUT_CHANNELS=[16,1]
_CN.NETWORK.CONVNET4D.PADDING=1
# #larger network
# _CN.NETWORK.CONVNET4D.KERNEL_SIZE=5 
# _CN.NETWORK.CONVNET4D.OUT_CHANNELS=[16,16,1]
# _CN.NETWORK.CONVNET4D.PADDING=2 


def get_cfg():
    "return a yacs cfgNode object with default values"
    cn =  _CN.clone()
    cn = lower_config(cn)
    return cn
    

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}