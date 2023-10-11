#!/usr/bin/env python3

"""
Author: Tomás Bizet de Barros
DRE: 116183736
"""

import numpy as np
import pandas as pd
import re
import math


def tokenize(s, separators):
    pattern = "|".join(map(re.escape, separators))
    tokens = re.split(pattern, s)
    if tokens[-1] == "":
        tokens.pop()

    return np.array([token for token in tokens if token])


def normalize(s):
    return s.lower().strip()


def remove_stopwords(tokens_list, stopwords):
    return [
        np.array([token for token in tokens if token not in stopwords])
        for tokens in tokens_list
    ]


def weigh_term(frequency, K, b, N, avg_doclen):
    return (
        ((K + 1) * frequency) / (K * ((1 - b) + b * (N / avg_doclen)) + frequency)
        if frequency > 0
        else 0
    )


def weigh_row(row, documents, term, K, b, N, avg_doclen):
    frequencies = np.array(
        list(map(lambda doc: np.count_nonzero(doc == term), documents))
    )
    weights = pd.Series(frequencies).apply(weigh_term, args=(K, b, N, avg_doclen))
    return weights


def generate_bm_matrix(documents, terms, K, b):
    bm_matrix = pd.DataFrame(index=terms, columns=range(len(documents)))
    N = len(documents)
    avg_doclen = np.mean(
        list(map(lambda doc: sum(map(lambda w: len(w), doc)), documents))
    )
    for term, row in bm_matrix.iterrows():
        bm_matrix.loc[term] = weigh_row(row, documents, term, K, b, N, avg_doclen)

    return bm_matrix


def similarity(document, query):
    # return document.dot(query) / (np.linalg.norm(document) * np.linalg.norm(query))
    # return document
    pass


def rank(documents, query, terms):
    pass
    # print(documents)
    # print(query, end="\n\n")
    # ranks = pd.Series(documents.index) in query
    # print(ranks)

    # def generate_query_series(word, terms):
    #     return 1 if word in terms else 0

    # query = query.apply(generate_query_series, args=(terms,))
    # print(query)
    # similarity(documents[1], query)
    # ranked_documents = (
    #     documents.apply(similarity, args=(query,)).T[0].sort_values(ascending=False)
    # )
    # return np.array(ranked_documents.index)


def main():
    # documentos
    dictionary = np.array(
        [
            "O peã e o caval são pec de xadrez. O caval é o melhor do jog.",
            "A jog envolv a torr, o peã e o rei.",
            "O peã lac o boi",
            "Caval de rodei!",
            "Polic o jog no xadrez.",
        ]
    )
    stopwords = ["a", "o", "e", "é", "de", "do", "no", "são"]  # lista de stopwords
    query = "xadrez peã caval torr"  # consulta
    separators = [" ", ",", ".", "!", "?"]  # separadores para tokenizacao

    # normalize
    normalized = np.array([normalize(s) for s in dictionary])
    # tokenize
    tokens_list = np.array([tokenize(s, separators) for s in normalized], dtype=object)
    # rmv stopwords
    tokens_list = remove_stopwords(tokens_list, stopwords)
    # terms
    terms = np.array(sorted(list(set([term for l in tokens_list for term in l]))))
    query = np.array(query.split())

    K = 1
    b = 0.75
    bm_matrix = generate_bm_matrix(tokens_list, terms, K, b)

    # query
    ranked_documents = rank(bm_matrix, query, terms)


if __name__ == "__main__":
    main()
