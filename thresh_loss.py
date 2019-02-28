import torch
import torch.nn.functional as F

def thresh_efficiency_loss(net):
    """
    Exploration encouraging component
    in loss functions. Encourages positive threshold
    """
    t_loss = 0
    for name,p in net.named_parameters():
        check = name.split('.')[-1]
        if check == 'thresh':
            t_loss += torch.pow(F.relu(-(p+1.0))+1.0,2) - (F.relu(p+1.0)-1.0)
    
    return t_loss