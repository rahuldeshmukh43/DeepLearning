import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, gt_label):
        """
        Args:
            x (Tensor) [B, num_classes]
            label (Tensor) [B] class index
        """
        return nn.CrossEntropyLoss()(x, gt_label)


def distance(x1, x2):
    """
    x1, x2 (Tensor) [B, d]
    """
    return torch.sum((x1 - x2) ** 2, dim=-1) 

class Center_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, center):
        """
        Args:
            x (Tensor) [B, d]
            center (Tensor) [B, d] -- class centers of x -- center.requires_grad is False
        Return
            distance (scalar) Mean Squared Distance between the class centers and x
        """
        d = torch.mean(distance(x, center)) # this is same as MSELoss
        return d

class Contrastive_Loss(nn.Module):
    def __init__(self, margin:float=1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, x1, x2, label1, label2):
        """
        Args:
            x1  (Tensor) [B, d]
            x2  (Tensor) [B, d]
            label1  (Tensor) [B]
            label2  (Tensor) [B]
        Return:
            loss (Tensor) scalar
        """
        indicator = (label1 == label2).to(torch.int32)
        d = distance(x1, x2) # [B]
        loss = indicator * d + (1 - indicator) * F.relu(self.margin - d) 
        return torch.mean(loss) # scalar

class Triplet_Loss(nn.Module):
    def __init__(self, 
                 margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_anchor_pos = distance(anchor, positive) #[B]
        d_anchor_neg = distance(anchor, negative) #[B]
        loss = F.relu( d_anchor_pos - d_anchor_neg + self.margin) #[B]
        return torch.mean(loss) # scalar
    
class Quadruplet_Loss(nn.Module):
    def __init__(self, 
                margin1: float = 1.0,
                margin2: float = 1.0) -> None:
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, pos, neg1, neg2):
        """
        Args:
            anchor (Tensor) [B, d] anchor sample
            pos  (Tensor) [B, d] positive sample from the anchor class
            neg1 (Tensor) [B, d] negative1 sample from a different class
            neg2 (Tensor) [B, d] negative2  from a third class (not anchor and not of neg1)
        """
        loss = F.relu(distance(anchor, pos) - distance(anchor, neg1) + self.margin1) #[B] same as triplet loss
        loss = loss + F.relu(distance(anchor, pos) - distance(neg1, neg2) + self.margin2) #[B] same as triplet loss
        return torch.mean(loss) #scalar
        
