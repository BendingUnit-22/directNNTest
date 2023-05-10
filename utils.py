import torch
import rbo

def lp_norms(deltas):
    l2_norms = []
    linfty_norms = []
    for delta in deltas:
        flattened_delta = torch.flatten(delta, start_dim=1)
        for single in flattened_delta:
            linfty_norms.append(torch.norm(single, p=torch.inf).numpy())
            l2_norms.append(torch.norm(single, p=2).numpy())
    return linfty_norms, l2_norms


def calc_similarity(idx1, idx2):
    similarity = []
    #print(idx1)
    for i in range(idx1.shape[0]): 
        rank1 = idx1[i].numpy()
        rank2 = idx2[i].numpy()
        similarity.append(rbo.RankingSimilarity(rank1, rank2).rbo(p=1))
        #similarity.append(None)
    return similarity

def weight_avg(nums, weights):
    if len(nums) != len(weights):
        print("Weighted average is used incorrectly")
        raise Exception
    score = 0.0
    for i, num in enumerate(nums):
        score += num * weights[i]
    return score/sum(weights)
