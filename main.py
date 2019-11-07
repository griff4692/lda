import os

import argparse
import numpy as np

from vocab import Vocab


def safe_multiply(a, b):
    return np.exp(np.log(a + 1e-5) + np.log(b + 1e-5))


def top_words(betas, vocab, n=10):
    for c in range(betas.shape[0]):
        proportions = betas[c, :]
        topk = (-proportions).argsort()[:n]
        topk_tokens = []
        for tok_id in topk:
            topk_tokens.append(vocab.get_token(tok_id))
        print('Topic {}: {}'.format(c + 1, ', '.join(topk_tokens)))


def create_vocab(vocab_fn):
    vocab_file = open(vocab_fn, 'r')
    vocab = Vocab()
    for token in vocab_file:
        token = token.strip()
        if len(token) > 0:
            vocab.add_token(token.strip())
    print('Added {} tokens to vocabulary'.format(vocab.size()))
    return vocab


def calculate_log_joint(args, data, betas, thetas, z, beta_prior, theta_prior):
    log_joint = 0.0
    for k in range(args.K):
        beta_lprob = 0.0  # TODO not working ~ np.log(dirichlet_pdf(betas[k], beta_prior))
        log_joint += beta_lprob

    for n in range(len(z)):
        # TODO ~ not working
        # theta_lprob = np.log(dirichlet.pdf(thetas[n], theta_prior))
        # log_joint += theta_lprob
        for m in range(len(z[n])):
            assignment = z[n][m]
            word = data[n][m]
            word_log_prob = np.log(betas[assignment][word])
            assignment_log_prob = np.log(thetas[n][assignment])
            log_joint += word_log_prob + assignment_log_prob
    return log_joint


def load_data(data_fn):
    raw_data = open(data_fn, 'r')
    data_dicts = []
    data = []

    for data_idx, line in enumerate(raw_data):
        split = line.strip().split()
        wc = {}
        doc_words = []
        wc['all'] = int(split[0])
        for i in range(1, len(split)):
            token_id, count = split[i].split(':')
            token_id = int(token_id)
            count = int(count)
            wc[token_id] = count
            for _ in range(count):
                doc_words.append(token_id)
        data_dicts.append(wc)
        data.append(doc_words)
    return data, data_dicts


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LDA with Gibbs Sampling')
    parser.add_argument('--data_dir', default='~/Desktop/ap/')
    parser.add_argument('--topic_words_dirichlet_prior', type=float, default=1.0)
    parser.add_argument('--topic_proportions_dirichlet_prior', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=10, help='Number of topics')

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)

    vocab_fn = os.path.join(args.data_dir, 'vocab.txt')
    vocab = create_vocab(vocab_fn)

    data_fn = os.path.join(args.data_dir, 'ap.dat')
    data, _ = load_data(data_fn)
    N = len(data)

    print('Loaded {} articles'.format(len(data)))

    V = vocab.size()
    # Initialize Beta, theta, and z
    # Beta ~ K x V drawn from Dirichlet
    betas = np.zeros([args.K, V])
    topic_words_dirichlet_prior_vec = np.zeros([V, ])
    topic_words_dirichlet_prior_vec.fill(args.topic_words_dirichlet_prior)
    betas = np.random.dirichlet(topic_words_dirichlet_prior_vec, size=args.K)
    print("Random Topics...")
    top_words(betas, vocab)

    topic_proportions_dirichlet_prior_vec = np.zeros([args.K, ])
    topic_proportions_dirichlet_prior_vec.fill(args.topic_proportions_dirichlet_prior)
    thetas = np.random.dirichlet(topic_proportions_dirichlet_prior_vec, size=N)

    # 2 z counts we need
    # for each document, what are the empirical topic assignments ~ N x K
    # for each topic, what are the empirical word assignments ~ K x V
    topic_assignments_for_docs = np.zeros([N, args.K])
    topic_assignments_for_words = np.zeros([args.K, V])

    z = []
    for n in range(N):
        M = len(data[n])
        z.append(np.random.randint(args.K, size=M))

    # update topic assignment proportions for documents
    for n in range(N):
        M = len(data[n])
        for word_idx, topic_assignment in zip(data[n], z[n]):
            topic_assignments_for_docs[n, topic_assignment] += 1

    max_iterations = 10000
    for iter_num in range(1, max_iterations + 1):
        for n, doc in enumerate(data):
            # sample topic proportions for document
            doc_topic_assignment_counts = topic_assignments_for_docs[n]
            new_doc_topic_proportions = np.random.dirichlet(
                list(map(lambda x: x + args.topic_proportions_dirichlet_prior, doc_topic_assignment_counts)))
            thetas[n, :] = new_doc_topic_proportions

            topic_assignments_for_docs[n] = 0  # Restart counter
            for m, token_id in enumerate(doc):
                # compute p(z=k|beta_k, theta_k, token_id) for each k
                topic_likelihoods = np.zeros([args.K, ])
                for k in range(args.K):
                    topic_likelihoods[k] = safe_multiply(new_doc_topic_proportions[k], betas[k][token_id])
                topic_likelihoods_norm = topic_likelihoods / topic_likelihoods.sum()
                new_z = np.random.choice(args.K, 1, p=topic_likelihoods_norm)[0]
                topic_assignments_for_docs[n, new_z] += 1
                topic_assignments_for_words[new_z, token_id] += 1
                z[n][m] = new_z

        for k in range(args.K):
            new_topic_word_proportions = np.random.dirichlet(
                list(map(lambda x: x + args.topic_words_dirichlet_prior, topic_assignments_for_words[k])))
            betas[k, :] = new_topic_word_proportions

        log_joint = calculate_log_joint(args, data, betas, thetas, z, topic_words_dirichlet_prior_vec,
                                        topic_proportions_dirichlet_prior_vec)
        print('Finished with iteration {}. Log joint {}'.format(iter_num, log_joint))
        # top_words(betas, vocab)
    print('All done!')
