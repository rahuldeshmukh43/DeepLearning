import torch
import torch.nn as nn

class NCNet_Loss(nn.Module):
    def __init__(self): #, net):
        super(NCNet_Loss, self).__init__()
        # self.net = net
        
    # def compute_mean_matching_scores(self, correlation):
    #     self.net.ComputeAndDetect(correlation)
    #     mean_score_A, mean_score_B = self.net._compute_mean_scores()
    #     return mean_score_A, mean_score_B
        
    def forward(self, correlation, gt_label, mean_score_A, mean_score_B):
        # mean_score_A, mean_score_B  = self.compute_mean_matching_scores(correlation)
        loss = -1.0*gt_label*(mean_score_A + mean_score_B)
        loss = torch.mean(loss)
        return loss