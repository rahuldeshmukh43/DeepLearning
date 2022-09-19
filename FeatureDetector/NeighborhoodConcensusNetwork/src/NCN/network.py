import torch 
import torch.nn as nn
import einops

from src.backbone.resnet import ResNet
from src.NCN.conv4d import Conv4d, BatchNorm4D


def set_requires_grad(network: nn.Module, layer_name: str = '', set_to: bool = False ):
    if layer_name != '':
        for param in getattr(network, layer_name).parameters():
            param.requires_grad = set_to
    else:
        for param in network.parameters():
            param.requires_grad = set_to
    return

class NCNet(nn.Module):
    def __init__(self, args, config, device):
        super(NCNet, self).__init__()
        self.args = args
        self.device = device        
        self.backbone = ResNet(encoder=config['network']['backbone'], 
                               pretrained=config['network']['pretrained']).to(self.device)
        #fixing the backbone
        if self.args.fix_backbone:
            for param in self.backbone.parameters(): #self.backbone.parameters():
                param.requires_grad= False
        else:
            if self.args.train_backbone_layer == '':
                pass
                # all params are trainable by default
            elif self.args.train_backbone_layer == 'layer3':
                set_requires_grad(self.backbone, layer_name='firstconv', set_to=False)
                set_requires_grad(self.backbone, layer_name='firstbn', set_to=False)
                set_requires_grad(self.backbone, layer_name='layer1', set_to=False)
                set_requires_grad(self.backbone, layer_name='layer2', set_to=False)
            elif self.args.train_backbone_layer == 'layer2':
                set_requires_grad(self.backbone, layer_name='firstconv', set_to=False)
                set_requires_grad(self.backbone, layer_name='firstbn', set_to=False)
                set_requires_grad(self.backbone, layer_name='layer1', set_to=False)
                # set_requires_grad(self.backbone, layer_name='layer2', set_to=False)
            elif self.args.train_backbone_layer == 'layer1':
                set_requires_grad(self.backbone, layer_name='firstconv', set_to=False)
                set_requires_grad(self.backbone, layer_name='firstbn', set_to=False)
                # set_requires_grad(self.backbone, layer_name='layer1', set_to=False)
                # set_requires_grad(self.backbone, layer_name='layer2', set_to=False)
            else:
                raise ValueError('%s is not a valid layer name for %s'%(self.args.train_backbone_layer,
                                                                        config['network']['backbone']))


        self.MNN = SoftMNN()
        self.Corr = Correlation()
        self.convNet4D = ConvNet4D(config['network']['convnet4d']).to(self.device)

    def forward(self, imA, imB, eps=1e-6):
        torch.autograd.set_detect_anomaly(True)
        feat_map_A = self.backbone(imA)
        feat_map_B = self.backbone(imB)
        corr = self.Corr(feat_map_A, feat_map_B)
        #compute NCNet output in both directions A->B and B->A
        corrA2B = self.convNet4D(self.MNN(corr).unsqueeze(1)).squeeze(1) #bijkl
        corrB2A = self.convNet4D(self.MNN(self._transpose(corr)).unsqueeze(1)).squeeze(1) #bklij
        out = (corrA2B + self._transpose(corrB2A))/2.
        out = self.MNN(out)
        #normalize final output to 0-1
        # non differentiable min, max
        # out_min,out_max = self._compute_max(out)
        #differentiable min, max
        out_min = out.min(1, keepdim=True)[0]
        out = out - out_min
        out_max = out.max(1, keepdim=True)[0] + eps
        out = out/out_max

        #changes made for dataparallel
        self.ComputeAndDetect(out)
        mean_score_A, mean_score_B = self._compute_mean_scores()
        return out, self.matches, self.score_A, self.score_B,  mean_score_A, mean_score_B

    @staticmethod
    def _compute_max(corr):
        """
        :param corr: correlation tensor [B,S,S,S,S] S=25
        """
        with torch.no_grad():
            min = corr.min(1, keepdim=True)[0]
            max = corr.max(1, keepdim=True)[0]
        return min, max

    def _transpose(self, corr):
        """
        take transpose of 4D tensor corr
        ijkl->klij and do this for each sample in the batch
        """
        return einops.rearrange(corr,'b i j k l -> b k l i j')
    
    def _ComputeScore(self, corr, dim:tuple, eps=1e-4):
        "compute matching score using softmax"
        assert len(dim) == 2
        #exponentiate all
        exp_corr = torch.exp(corr)
        # print('debug exponential infs',torch.any(torch.isinf(exp_corr)), torch.any(torch.isnan(exp_corr)))
        #compute sum along dims and normalize
        exp_sum = torch.sum(exp_corr, dim=dim, keepdims=True)
        exp_corr = exp_corr/(exp_sum + eps)
        return exp_corr

    def ComputeAndDetect(self, corr):
        "do hard assignment of correlation and give matching pixel location as output"
        self.score_A = self._ComputeScore(corr, (1,2))
        self.score_B = self._ComputeScore(corr, (3,4))
        #detect matches using scores
        self.matches = self.DetectMatches(self.score_B)
        return 
        
        # #compute mean scores
        # mean_score_A, mean_score_B = self._compute_mean_scores()
        # #output mean scores and matches
        # return mean_score_A, mean_score_B
    
    def _compute_mean_scores(self):
        B,H1,W1,_= self.matches.shape
        device = self.matches.device
        mean_score_A = torch.zeros(B, device=device)
        mean_score_B = torch.zeros(B, device=device)
        for b in range(B):
            for i in range(H1):
                for j in range(W1):
                    #k,l = [int(m.item()) for m in self.matches[b,i,j]]
                    k, l = self.matches[b, i, j].int()
                    #print(b,i,j,k,l)
                    mean_score_A[b]+= self.score_A[b,i,j,k,l]
                    mean_score_B[b]+= self.score_B[b,i,j,k,l]
        mean_score_A /= H1*W1
        mean_score_B /= H1*W1
        return mean_score_A, mean_score_B
    
    @torch.no_grad()
    def DetectMatches(self, score_B):
        """
        hard assignment matches using scores_B
        finds all matches (i,j),(k,l) for all possible (i,j) in first image  
        (k,l) = argmax_cd P( K=c, L=d | I=i, J=j)
              = argmax_cd s_{ijcd}^B
        """
        B,H1,W1,H2,W2 = score_B.shape
        device = score_B.device
        matches = torch.empty((B,H1,W1,2), device=device)#.to(self.device)
        for b in range(B):
            for i in range(H1):
                for j in range(W1):
                    m = torch.argmax(score_B[b,i,j,:,:])#.item() #m= k*num_cols + l
                    k = torch.div(m, W2, rounding_mode='floor').int() #int(m//W2) # row y
                    l = (m - k*W2).int() #col x
                    matches[b,i,j,:] = torch.tensor([k,l])
        return matches
    
    def test(self, imA, imB):
        corr = self.forward(imA, imB)
        #score_A = self._ComputeScore(corr, (1,2))
        self.score_B = self._ComputeScore(corr, (3,4))
        #detect matches using scores
        self.matches = self.DetectMatches(self.score_B)        
        return self.matches, self.score_B  
        
        
class Correlation(nn.Module):
    """ compute 4d dense correlation tensor using feature maps"""
    def __init__(self):
        super(Correlation, self).__init__()
        
    def forward(self, featA, featB):
        _B,_C,H1,W1 = featA.shape
        _,_,H2,W2 = featB.shape
        #compute dot product
        corr = torch.einsum('bcij,bckl->bijkl',featA, featB) #B,H1,W1,H2,W2
        #compute norms along feature dimension
        norm_featA = torch.norm(featA, dim=1) #B,h1,w1
        norm_featB = torch.norm(featB, dim=1) #B,h2,w2
        #repeat for division
        #norm_featA = norm_featA.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,H2,W2) #B,H1,W1,H2,W2
        #norm_featB = norm_featB.unsqueeze(1).unsqueeze(1).repeat(1,H1,W1,1,1) #B,H1,W1,H2,W2 
        norm_featA = einops.repeat(norm_featA, 'b h1 w1 -> b h1 w1 h2 w2', h2=H2, w2=W2)
        norm_featB = einops.repeat(norm_featB, 'b h2 w2 -> b h1 w1 h2 w2', h1=H1, w1=W1)
        #scale dot product to get cosine similarity
        corr /= norm_featA*norm_featB
        return corr
    
class SoftMNN(nn.Module):
    """ compute soft mutual nearest neighbor given a 4d correlation tensor"""
    def __init__(self):
        super(SoftMNN, self).__init__()
        
    def forward(self, corr):
        #compute scaling factors along ij and kl respectively
        #we divide by max for global scaling 
        # this essentially helps in finding the strongest matches globally 
        scale_for_A, scale_for_B = self._compute_scaling_terms(corr) # both of sizes B,H1,W1,H2,W2
        corr = corr*scale_for_A*scale_for_B
        return corr
    
    @torch.no_grad()
    def _compute_scaling_terms(self, corr, eps=1e-6):
        _B,H1,W1,H2,W2 = corr.shape
        max_A = einops.reduce(corr,'b i j k l -> b k l', 'max')
        max_A = einops.repeat(max_A, 'b k l -> b i j k l', i = H1, j = W1) + eps
        max_B = einops.reduce(corr,'b i j k l -> b i j', 'max')
        max_B = einops.repeat(max_B, 'b i j -> b i j k l', k = H2, l = W2) + eps
        scale_for_A = corr/max_A
        scale_for_B = corr/max_B
        return scale_for_A, scale_for_B
    
# convnet_conf = {'kernel_size':5, #3
#                 'out_channels':[16,16,1], #[16,1]
#                 'padding':2, #1
#                 }
    
class ConvNet4D(nn.Module):
    """ transform a 4d correlation tensor using 4d convs"""
    def __init__(self, conf):
        super(ConvNet4D, self).__init__()
        self.conf = conf
        self.kernel_size = 4*(self.conf['kernel_size'], ) #either 5 or 3
        self.out_channels = self.conf['out_channels'] #[16,16,1] or [16,1]
        self.padding = self.conf['padding']
        self.conv_layers = nn.ModuleList()
        in_channel = 1
        for out_channel in self.out_channels:
            conv_layer = ConvLayer(in_channel, 
                                   out_channel,
                                   self.kernel_size,
                                   self.padding,
                                   bias = False)
            self.conv_layers.append(conv_layer)
            in_channel= out_channel        
        
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)        
        return x
    
class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = Conv4d(in_channel, 
                           out_channel,
                           kernel_size,
                           padding = padding,
                           bias=bias)
        self.BN = BatchNorm4D(out_channel)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.BN(x)
        return torch.nn.ReLU()(x)        
    
    
    
# -----------------------------------------------------------------------------
# TESTING NETWORK ON IPYTHON
# -----------------------------------------------------------------------------
# from src.NCN.NCN_config import get_cfg, lower_config
# config = get_cfg()#
# _config = lower_config(config)#
# ncn_net = NCNnet(_config)
#
# import torch
## img1 = torch.rand(400,400,3)
## img2 = torch.rand(400,400,3)
## img1-=0.5
## img2-=0.5
#
# import einops
# img1 = einops.rearrange(img1,'h w c -> c h w').unsqueeze(0)
# img1.shape
# img2 = einops.rearrange(img2,'h w c -> c h w').unsqueeze(0)
#
# corr = ncn_net(img1,img2)
# m,ma, mb = ncn_net.ComputeAndDetect(corr)
## print stuff -- size checks
# matches
# m
# ma.shape
# ma.requires_grad
# m.requires_grad
# m
# m.shape
# m,ma, mb = ncn_net.ComputeAndDetect(corr)
# m,ma, mb = ncn_net.ComputeAndDetect(corr)
