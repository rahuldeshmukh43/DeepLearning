#!/bin/python
import configargparse

def get_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True, help='config file path')
    
    #PATH OPTIONS: datadir, logdir, outdir, ckpt_path
    parser.add_argument('--datadir', type=str, help='the dataset directory')
    parser.add_argument("--logdir", type=str, default='./logs/', help='dir of tensorboard logs')
    parser.add_argument("--outdir", type=str, default='./out/', help='dir of output e.g., ckpts')
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific checkpoint path to load the model from, '
                             'if not specified, automatically reload from most recent checkpoints')
    
    #GENERAL OPTIONS: experiment name, num iters, mode
    parser.add_argument("--exp_name", type=str, help='experiment name')
    #parser.add_argument('--n_iters', type=int, default=2000, help='max number of training iterations')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--phase', type=str, default='train', help='train/test')#, choice=['train', 'test'])
    parser.add_argument('--multi_gpu', action='store_true', help='flag for muti-gpu training')
    parser.add_argument('--debug', action='store_true', help='flag for debugging code')
    
    #DATA OPTIONS: workers
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--split', nargs=3, type=int, help='ratio of data split train,val,test', default=[700,300,300])
    
    #TRAINING OPTIONS: batch_size, learning rate, scheduler params
    parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
    
    #MODEL OPTIONS: backbone, model configs, pretrained etc
    parser.add_argument('--fix_backbone', action='store_true', help='flag for fixing wts of backbone')
    parser.add_argument('--train_backbone_layer', type=str, default='' , help=' layer name of backbone to be trained. '
                                                                              'Default empty string trains entire backbone'
                                                                              ' also accepts layer1, layer2, layer3 as inputs')
    #  handled by yacs config
    # parser.add_argument('--backbone', type=str, default='resnet50',
    #                     help='backbone for feature representation extraction. supported: resent')
    # parser.add_argument('--pretrained', type=int, default=1,
    #                     help='if use ImageNet pretrained weights to initialize the network')
    
    #LOSS FUNCTION OPTIONS
    # no config for loss
    
    #LOGGING OPTIONS: logging intervals, wt saving intervals
    parser.add_argument('--log_scalar_interval', type=int, default=20, help='print interval')
    parser.add_argument('--log_scalar_interval_eval', type=int, default=20, help='print interval for validation')
    parser.add_argument('--log_img_interval', type=int, default=100, help='log image interval')
    parser.add_argument("--save_interval", type=int, default=100, help='frequency of weight ckpt saving')
    parser.add_argument("--log_name", type=str, default='training', help='base file name for logging training progress')
    
    #EVAL OPTIONS: input and output dir for eval
    parser.add_argument('--extract_img_dir', type=str, help='the directory of images to extract features')
    parser.add_argument('--extract_out_dir', type=str, help='the directory of images to extract features')
    
    args = parser.parse_args()
    
    return args
