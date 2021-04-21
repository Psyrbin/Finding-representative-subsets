from sklearn.feature_selection import VarianceThreshold
from fast_pytorch_kmeans import KMeans
from scipy.stats import norm
import numpy as np
from itertools import combinations_with_replacement

def variancedist(W, S):
    try:
        R = np.linalg.inv(S.T @ S)
    except:
        # S += np.eye(S.shape[1]) * 0.001
        R = np.linalg.inv(S.T @ S + np.eye(S.shape[1]) * 0.001)
    D = np.sum( (W @ R) * W, axis=1)
    return D

def dopt(topicmat, k):
    index = [np.random.choice(topicmat.shape[0])] # in R script it samples from shape[1], but then uses it to index rows, so I thought it should sample from row count
    rows = np.array(range(topicmat.shape[0]))
    rows = np.delete(rows, index)
    S = topicmat[index,:]
    # S = topicmat.iloc[index,:] #if it's dataframe
    W = topicmat[~np.isin(range(topicmat.shape[0]), index)]
    # print(S.shape, W.shape)

    while len(index) < k:
        i = np.argmax(variancedist(W, S))
        S = np.vstack((S, W[i,:])) #np.append(S.append(W[i,:]) # S=
        W = np.delete(W, i, axis=0) #W.drop(i, inplace=True)
        index.append(rows[i])
        rows = np.delete(rows, i)
        # print(S.shape, W.shape)

    return index

def random(data, num_obs):
    return np.random.choice(data.shape[0], num_obs, replace=False)

def kmeans(obs, num_obs):
    kmeans = KMeans(n_clusters=num_obs, mode='euclidean', verbose=1)
    labels = kmeans.fit_predict(obs)
    label_idx_dict = {}
    for index, label in enumerate(labels):
        label = label.item()
        if label in label_idx_dict:
            label_idx_dict[label].append(index)
        else:
            label_idx_dict[label] = [index]
    indices = [choices(label_idx_dict[key], k=1)[0] for key in label_idx_dict]
    if len(indices) < num_obs:
        more_indices = [choices(label_idx_dict[key], k=1)[0] for key in label_idx_dict]
        for idx in more_indices:
            if idx not in indices:
                indices.append(idx)
            if len(indices) == num_obs:
                break
    return indices

def variance(obs, num_obs):
    selector = VarianceThreshold()
    selector.fit_transform(data.transpose())
    indices = np.argsort(selector.variances_)[-num_obs:]
    return indices

def fit_norm(obs):
    mu_list = []
    sd_list = []
    n_obs = obs.shape[0]
    for i in range(n_obs):
        mu, sd = norm.fit(obs[i,:])
        mu_list.append(mu)
        sd_list.append(sd)
    return mu_list, sd_list

def gaussian_kld(mu1, sd1, mu2, sd2):
    return np.log(sd2/sd1) + ((sd1**2 + (mu1-mu2)**2) / (2*(sd2**2))) - 0.5

def kld_matrix(mu_list, sd_list, dataset_name="", embed_type=""):
    dshape = len(mu_list)
    kld_matrix = np.zeros((dshape, dshape))
    looper = combinations_with_replacement(range(dshape), 2)
    for i, j in looper:
        kld_ij = gaussian_kld(mu_list[i], sd_list[i], mu_list[j], sd_list[j]) + gaussian_kld(mu_list[j], sd_list[j], mu_list[i], sd_list[i])
        kld_matrix[i][j] = kld_ij
        kld_matrix[j][i] = kld_ij
    print('Saving kld matrix...')
    np.save(dataset_name+'_kld_'+embed_type, kld_matrix)
    #return kld_matrix
    return kld_matrix


def furthestPointSampler(kld_matrix, num_obs):
    indices = np.zeros(num_obs, dtype=np.int64)
    # select two farthest points
    indices[0], indices[1] = np.unravel_index(kld_matrix.argmax(), kld_matrix.shape)
    for i in range(2, num_obs):
        # maximize minimum distance to all points in indices
        sorted_indices = np.argsort(np.min(kld_matrix[indices[:i],:], axis=0))[::-1]
        #sorted_indices = np.setdiff1d(sorted_indices, indices[:i])
        sorted_indices = sorted_indices[~np.in1d(sorted_indices, indices[:i])]
        indices[i] = sorted_indices[0]
    return indices

def kld(data, num_obs):
    mu_list, sd_list = fit_norm(data)
    matrix = kld_matrix(mu_list, sd_list)
    return furthestPointSampler(matrix, num_obs)
