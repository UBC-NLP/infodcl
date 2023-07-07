import torch
import torch.nn as nn
from torch.nn import functional as F
from pprint import pprint

### Credits https://github.com/HobbitLong/SupContrast
###         https://github.com/varsha33/LCL_loss

class Soft_SupConLoss_PMI(nn.Module):

    def __init__(self, weights, num_classes, temperature=0.07, device='cpu'):
        super(Soft_SupConLoss_PMI, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.device = device

        self.weights = weights.to(self.device)


    def forward(self, features, labels=None, mask=None):
        """
        Returns:
            A loss scalar.
        """
        features = F.normalize(features, dim=1, p=2)

        batch_size = features.shape[0]

        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)


        contrast_feature = features
        anchor_feature = contrast_feature

        # compute dot product of embeddings
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), 
            self.temperature)

        # set diagonal as 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )

        ## it produces 0 for the non-matching places and 1 for matching places and neg mask does the opposite
        ## set diagonal as 0
        mask = mask * logits_mask

        weighted_mask = torch.matmul(torch.matmul(labels_one_hot, self.weights), torch.transpose(labels_one_hot, 0, 1)).to(self.device)

        
        #remove diagonal
        weighted_mask = weighted_mask * logits_mask

        # weights of postive samples
        pos_weighted_mask = weighted_mask * mask

        # compute log_prob with logsumexp
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        # remove diagonal
        logits = anchor_dot_contrast - logits_max.detach()

        # wiyk * exp(hi * hk / t)
        exp_logits = torch.exp(logits) * weighted_mask

        ## log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_weighted_mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()
        # print("loss", loss)
        return loss


class Soft_SupConLoss_CLS(nn.Module):

    def __init__(self, num_classes, temperature=0.07, device='cpu'):
        super(Soft_SupConLoss_CLS, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.device = device

    def forward(self, features, labels=None, weights=None, mask=None):
        """
        Returns:
            A loss scalar.
        """

        features = F.normalize(features, dim=1, p=2)

        batch_size = features.shape[0]
        weights = F.softmax(weights,dim=1) # logit to softmax

        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_feature = features
        anchor_feature = contrast_feature

        # compute dot product of embeddings
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), 
            self.temperature)

        # set diagonal as 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )

        ## it produces 0 for the non-matching places and 1 for matching places and neg mask does the opposite
        ## set diagonal as 0
        mask = mask * logits_mask

        weighted_mask = torch.matmul(weights, torch.transpose(labels_one_hot, 0, 1)).to(self.device)

        weighted_mask = weighted_mask * logits_mask

        # weights of postive samples
        pos_weighted_mask = weighted_mask * mask

        # compute log_prob with logsumexp
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        # remove diagonal
        logits = anchor_dot_contrast - logits_max.detach()

        # wiyk * exp(hi * hk / t)
        exp_logits = torch.exp(logits) * weighted_mask

        ## log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_weighted_mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        # print("loss",loss)
        return loss

