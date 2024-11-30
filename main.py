import numpy as np
import pickle

from load_data import load_data
from utils import create_hash_id_tables, get_doc_vecs, approximate_knn, get_doc_embedding

import os
from dotenv import load_dotenv
load_dotenv()


def find_similar_docs(doc, num_universes = 5, num_similar = 3):
    N_dim = int(os.getenv('N_dim'))
    N_planes = int(os.getenv('N_planes'))
    N_universes = int(os.getenv('N_universes'))

    planes_l = []
    for _ in range(N_universes): 
        planes_l.append(np.random.normal(size = (N_dim, N_planes)))

    all_tweets = load_data()
    en_embeddings = pickle.load(open('./en_embeddings.p', 'rb'))
    document_vecs, ind2Tweet = get_doc_vecs(all_tweets, en_embeddings)

    hash_tables, id_tables = create_hash_id_tables(document_vecs, planes_l, N_universes)

    nearest_nbr_ids = approximate_knn(doc, get_doc_embedding(doc, en_embeddings), planes_l, hash_tables, id_tables, num_universes_to_use = num_universes, k = num_similar)

    print('original tweeet:', doc)
    for neighbor_id in nearest_nbr_ids:
        print(f"Nearest neighbor at document id {neighbor_id}")
        print(f"document contents: {all_tweets[neighbor_id]}")