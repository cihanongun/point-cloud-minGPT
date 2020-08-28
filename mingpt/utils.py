import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
        
def plotPC(pcList, show = True, save = False, name=None, figCount=9 , sizex = 12, sizey=3):
    
    '''
    Function for visualizing multiple point cloud models. It uses color values if available.
    You can feed more than one point cloud array as a list to plot them as multiple lines.
    '''
    
    if (len(np.shape(pcList)) == 3) :  pcList = [pcList] # if single array
    listCount = len(pcList)
    pIndex = 1
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for l in range(listCount):
        pc = pcList[l]
        
        for f in range(figCount):

            ax = fig.add_subplot(listCount, figCount, pIndex, projection='3d')
        
            if(np.shape(pcList[0])[2] == 4): # colors
                c_values = [colors[x-3] for x in pc[f,:,3].astype(int)]
            else:
                c_values = 'b'
            
            ax.scatter(pc[f,:,0], pc[f,:,2], pc[f,:,1], c=c_values, marker='.', alpha=0.8, s=8)

            ax.set_xlim3d(0.25, 0.75)
            ax.set_ylim3d(0.25, 0.75)
            ax.set_zlim3d(0.25, 0.75)
            
            plt.axis('off')
            
            pIndex += 1
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig