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


def weigh_term(frequency, frequency_in_collection, N):
    return (
        1 + np.log2(frequency) * np.log2(N / frequency_in_collection)
        if frequency > 0
        else 0
    )


def weigh_row(row, documents, term):
    frequencies = np.array(
        list(map(lambda doc: np.count_nonzero(doc == term), documents))
    )
    frequency_in_collection = np.count_nonzero(np.concatenate(documents) == term)
    N = len(row.index)
    weights = pd.Series(frequencies).apply(
        weigh_term, args=(frequency_in_collection, N)
    )
    return weights


def generate_tfidf_matrix(documents, terms):
    tfidf_matrix = pd.DataFrame(index=terms, columns=range(len(documents)))
    for term, row in tfidf_matrix.iterrows():
        tfidf_matrix.loc[term] = weigh_row(row, documents, term)

    return tfidf_matrix


def similarity(document, query):
    return document.dot(query) / (np.linalg.norm(document) * np.linalg.norm(query))


def rank(documents, query):
    ranked_documents = (
        documents.apply(similarity, args=(query,)).T[0].sort_values(ascending=False)
    )
    return np.array(ranked_documents.index)


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
    query = np.array([query.split()])

    tfidf_matrix = generate_tfidf_matrix(tokens_list, terms)
    query_weights = generate_tfidf_matrix(query, terms)
    ranked_documents = rank(tfidf_matrix, query_weights)
    print(ranked_documents)


if __name__ == "__main__":
    main()
