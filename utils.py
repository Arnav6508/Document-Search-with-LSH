import numpy as np
from preprocess import preprocess

def cosine_similarity(u, v):
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def knn(vec_Y, Y, k = 1):
    similarity = []
    for Y_ele in Y: similarity.append(cosine_similarity(vec_Y, Y_ele))

    sorted_ids = np.argsort(np.array(similarity))[::-1]
    return sorted_ids[:k]


def get_doc_embedding(tweet, en_embeddings):
    tweet = preprocess(tweet)
    doc_embedding = np.zeros(300)
    for word in tweet: doc_embedding += en_embeddings.get(word,0)
    return doc_embedding


def get_doc_vecs(all_docs, en_embeddings):
    all_docs_embedding = []
    ind2doc = {}

    for i, doc in enumerate(all_docs): 
        doc_embedding = get_doc_embedding(doc, en_embeddings)
        all_docs_embedding.append(doc_embedding)
        ind2doc[i] = doc_embedding

    doc_vec_matrix = np.vstack(all_docs_embedding)

    return doc_vec_matrix, ind2doc


def hash_value_of_vector(v,planes):
    '''
    dim(v)      = (1, N_dim)
    dim(planes) = (N_dim, N_planes)
    '''

    sign_vec = np.sign(np.dot(v, planes)) # will contain 1s and -1s
    sign_vec_bool = sign_vec>=0           # will contain 1s and  0s
    sign_vec_bool = np.squeeze(sign_vec_bool)

    n_planes = 10

    hash_val = 0
    for i in range(n_planes): 
        hash_val += (2**i)*sign_vec_bool[i]

    return int(hash_val)

def make_hash_table(vecs, planes):
    hash2vec, hash2id = {}, {}

    for i, vec in enumerate(vecs):
        hash_val = hash_value_of_vector(vec,planes)

        if (hash2vec.get(hash_val,[]) == []): hash2vec[hash_val] = [vec]
        else: hash2vec[hash_val].append(vec)

        if (hash2id.get(hash_val,[]) == []): hash2id[hash_val] = [i]
        else: hash2id[hash_val].append(i)

    return hash2vec, hash2id

def create_hash_id_tables(vecs, planes_l, N_universes):
    hash_tables, id_tables = [], []

    for universe_id in range(N_universes):
        hash2vec, hash2id = make_hash_table(vecs, planes_l[universe_id])
        hash_tables.append(hash2vec)
        id_tables.append(hash2id)
    
    return hash_tables, id_tables


def approximate_knn(doc, vec, planes_l, hash_tables, id_tables, num_universes_to_use, k = 1):
    docs_to_consider, id_to_consider = [], []
    new_id_to_consider = set()

    for universe_id in range(num_universes_to_use):

        planes = planes_l[universe_id]
        hash_table = hash_tables[universe_id]
        id_table = id_tables[universe_id]

        hash_val = hash_value_of_vector(vec,planes)
        doc_vecs = hash_table[hash_val]

        for i, new_id in enumerate(id_table[hash_val]):

            if new_id in new_id_to_consider: continue

            docs_to_consider.append(doc_vecs[i])
            id_to_consider.append(new_id)
            new_id_to_consider.add(new_id)

    docs_to_consider = np.array(docs_to_consider)
    nearest_neighbours_idx = knn(vec, docs_to_consider, k = k)
    nearest_neighbours_id = [id_to_consider[idx] for idx in nearest_neighbours_idx]

    return nearest_neighbours_id