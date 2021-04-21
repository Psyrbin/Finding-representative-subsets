import argparse
import pandas as pd
import numpy as np

import featurizers
import observation_selectors

parser = argparse.ArgumentParser()

parser.add_argument('--featurizer', default=None, help='Available options: cvec, tfidf, roberta, bert, distilroberta, glove, universal, tsne')
parser.add_argument('--selector', default='random', help='Available options: dopt, random, kmeans, variance, kld')
parser.add_argument('--n_observations', default=100, type=int)
parser.add_argument('--output', default='observations.npy')
parser.add_argument('--use_precomputed_features', default=False, type=bool)
parser.add_argument('--input_path', default=None)

args = parser.parse_args()

# Featurize
if args.use_precomputed_features:
    features = pd.read_csv(args.input_path)
else:
    data = pd.read_csv(args.input_path)['text']
    if args.featurizer == 'cvec':
        features = featurizers.cvec(data)
    elif args.featurizer == 'tfidf':
        features = featurizers.tfidf(data)
    elif args.featurizer == 'roberta':
        features = featurizers.roberta(data)
    elif args.featurizer == 'bert':
        features = featurizers.bert(data)
    elif args.featurizer == 'distilroberta':
        features = featurizers.distilroberta(data)
    elif args.featurizer == 'glove':
        features = featurizers.glove(data)
    elif args.featurizer == 'universal':
        features = featurizers.universal(data)
    elif args.featurizer == 'tsne':
        features = featurizers.tsne(data)
    else:
        print('Invalid featurizer')
        import sys
        sys.exit()

    print('Featurization complete')

# Select observations
if args.selector == 'dopt':
    indices = observation_selectors.dopt(features, args.n_observations)
elif args.selector == 'random':
    indices = observation_selectors.random(features, args.n_observations)
elif args.selector == 'kmeans':
    indices = observation_selectors.kmeans(features, args.n_observations)
elif args.selector == 'variance':
    indices = observation_selectors.variance(features, args.n_observations)
elif args.selector == 'kld':
    indices = observation_selectors.kld(features, args.n_observations)
else:
    print('Invalid selector')
    import sys
    sys.exit()

np.save(args.output, indices)
