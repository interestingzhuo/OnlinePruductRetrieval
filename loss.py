import torch.nn as nn
import torch
class TripletLoss(nn.Module):

    def __init__(self, p, k, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.batch_p = p
        self.batch_k = k
        self.create_mask()
    def create_mask(self):
        w = self.batch_p*self.batch_k
        mask = torch.zeros(w,w)
        pos = torch.ones(self.batch_k,self.batch_k)
        for i in range(self.batch_p):
            mask[i*self.batch_k:(i+1)*self.batch_k,i*self.batch_k:(i+1)*self.batch_k] = pos   
        min_mask = mask.cuda()
        max_mask = (1-min_mask)*-1000
        self.max_mask = max_mask.type(torch.cuda.FloatTensor)
        self.min_mask = min_mask.type(torch.cuda.FloatTensor)
        
    def forward(self, output, lbls):
        tripletloss = self.triplet_loss_batch(output,self.batch_p,self.batch_k,self.min_mask,self.max_mask,margin=self.margin)
        return tripletloss
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


    def euclid_dist_TF(self,A, B):	
        M = torch.mm(A, B.transpose(1,0))
        H = torch.sum(torch.mul(A,A), dim=1,keepdim = True)
        K = torch.sum(torch.mul(B,B), dim=1,keepdim = True)
        D = -2*M + H + K.transpose(1,0)
        return D


    def triplet_loss_batch(self, x,batch_p,batch_k,min_mask,max_mask,margin=0.1):
        """Add L2Loss to all the trainable variables.
        
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits is output of AlexNet model.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        D = self.euclid_dist_TF(x, x).cuda()
        D = D.type(torch.cuda.DoubleTensor)
        Dap = torch.max((D + max_mask),dim=1)[0]
        Dan = torch.min((D + min_mask), dim=1)[0]
        Dx = (Dap+margin-Dan).type(torch.cuda.FloatTensor)
        zero = torch.zeros(Dx.shape[0]).cuda()
        trip_loss = torch.max(Dx,zero)
        trip_loss = torch.mean(trip_loss)
        return trip_loss


class ContrastiveLoss(nn.Module):

    def __init__(self, p, k,margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch_p = p
        self.batch_k = k
        self.create_mask()
    def create_mask(self):
        w = self.batch_p*self.batch_k
        mask = torch.zeros(w,w)
        pos = torch.ones(self.batch_k,self.batch_k)
        for i in range(self.batch_p):
            mask[i*self.batch_k:(i+1)*self.batch_k,i*self.batch_k:(i+1)*self.batch_k] = pos   
        min_mask = mask.cuda()
        max_mask = (1-min_mask)
        self.max_mask = max_mask.type(torch.cuda.FloatTensor)
        self.min_mask = min_mask.type(torch.cuda.FloatTensor)
        
    def forward(self, output1):
        contrastiveloss = self.contrastive_loss_batch(output1,self.batch_p,self.batch_k,self.min_mask,self.max_mask,margin=self.margin)
        return contrastiveloss
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

    def euclid_dist_TF(self,A, B):	
        M = torch.mm(A, B.transpose(1,0))
        H = torch.sum(torch.mul(A,A), dim=1,keepdim = True)
        K = torch.sum(torch.mul(B,B), dim=1,keepdim = True)
        D = -2*M + H + K.transpose(1,0)
        return D

    def contrastive_loss_batch(self, x,batch_p,batch_k,p_mask,n_mask,margin=0.1):
        """Add L2Loss to all the trainable variables.
        
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits is output of AlexNet model.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        D = self.euclid_dist_TF(x, x).cuda()
        Dp = D
        Dn = (torch.clamp(margin-D, min=0)).type(torch.cuda.FloatTensor)
        Dp = Dp.mul(p_mask)
        Dn = Dn.mul(n_mask)
        Dp = torch.max((Dp),dim=1)[0]
        Dn = torch.max((Dn),dim=1)[0]
        Dp = torch.mean(Dp)
        Dn = torch.mean(Dn)
        contrastive_loss = (Dp+Dn)/2

        return contrastive_loss



class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP, self).__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims
    def sigmoid(self, tensor, temp=1.0):
        """ temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y


    def compute_aff(self, x):
        """computes the affinity matrix between an input vector and itself"""
        return torch.mm(x, x.t())


    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size) 
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        
        sim_all = self.compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid and ignores the relevance score of the query to itself
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # compute the rankings,all batch
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = self.sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1-ap)


class CircleLoss(nn.Module):

    def __init__(self, p, k,gamma=80,margin=0.4):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.batch_p = p
        self.batch_k = k
        self.create_mask()
        self.gamma =gamma
    def create_mask(self):
        w = self.batch_p*self.batch_k
        mask = torch.zeros(w,w)
        pos = torch.ones(self.batch_k,self.batch_k)
        for i in range(self.batch_p):
            mask[i*self.batch_k:(i+1)*self.batch_k,i*self.batch_k:(i+1)*self.batch_k] = pos   
        neg_mask = mask.cuda()
        pos_mask = (1-mask).cuda()
        self.pos_mask = pos_mask.type(torch.cuda.FloatTensor)
        self.neg_mask = neg_mask.type(torch.cuda.FloatTensor)
        
    def forward(self, output1,output2, label):
        circleloss = self.circle_loss_batch(output1,self.pos_mask,self.neg_mask,self.gamma,margin=self.margin)
        return circleloss
    def circle_loss_batch(self,x,pos_mask,neg_mask,gamma,margin=0.1):
        """Add L2Loss to all the trainable variables.
        
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits is output of AlexNet model.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        S = torch.mm(x, x.transpose(1,0)).cuda()
        S = S.type(torch.cuda.DoubleTensor)
        Sap = torch.min((S + pos_mask*1000),dim=1)[0]
        San = torch.max((S + neg_mask*-1000), dim=1)[0]
        zero = torch.zeros(S.shape[0]).cuda().double()
        loss = torch.log(1+torch.exp(gamma*(torch.max(San+margin,zero)*(San-margin)))*torch.exp(-gamma*(torch.max(1-Sap+margin,zero)*(Sap+margin-1))))
        #loss = torch.log(1+torch.exp(gamma*(San-margin)**2)*torch.exp(gamma*((1-Sap)-margin)**2))
        loss = torch.mean(loss)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'




# class CircleLoss(nn.Module):
#     #def __init__(self, m: float, gamma: float) -> None:
#     def __init__(self, p, k,lamda,gamma=80,margin=0.4):
#         super(CircleLoss, self).__init__()
#         self.m = margin
#         self.gamma = gamma
#         self.soft_plus = nn.Softplus()#log(1+x)

#     def forward(self, feat1: Tensor,feat2: Tensor, lbl: Tensor) -> Tensor:
#         sp, sn = convert_label_to_similarity(feat1, lbl)
#         ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
#         an = torch.clamp_min(sn.detach() + self.m, min=0.)

#         delta_p = 1 - self.m
#         delta_n = self.m

#         logit_p = - ap * (sp - delta_p) * self.gamma
#         logit_n = an * (sn - delta_n) * self.gamma

#         loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
#         return loss
        
