from m1_main import *

import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import matplotlib.pyplot as plt

def compute_perplexity_values(dictionary, corpus, texts, limit, start=2, step=1):

    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                         update_every=0, passes=10, iterations=10, per_word_topics=True, minimum_probability=0.25)
        model_list.append(model)
        perplexity_values.append(model.log_perplexity(corpus))

    return model_list, perplexity_values

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                         update_every=0, passes=10, iterations=10, per_word_topics=True, minimum_probability=0.25)
        model_list.append(model)
        coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model_lda.get_coherence())

    return model_list, coherence_values

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':

  df = pd.read_pickle(lda_preprocessed_df_path)

  nested_tokens = [list(token_list) for token_list in df['Composite_tokens']]
  dictionary = Dictionary(nested_tokens)
  corpus = [dictionary.doc2bow(token_list) for token_list in nested_tokens]

  start = 2
  limit = 99
  step = 1

  # Compute perplexity scores ---------------------------------------------------------------------------------------------
  model_list, perplexity_values = compute_perplexity_values(dictionary=dictionary, corpus=corpus,
                                                            texts=nested_tokens, start=start, limit=limit, step=step)

  # Plot perplexity scores
  x = range(start, limit, step)
  plt.figure(figsize=(10,6), dpi=1200)
  plt.xlabel("Num Topics")
  plt.ylabel("Perplexity score")
  plt.legend(("perplexity_values"), loc='best')
  plt.xticks(x, rotation=90)
  plt.grid(True)
  plt.plot(x, perplexity_values)
  plt.tight_layout()
  plt.savefig(workflow_folder + r'\visuals\lda_perplexity.png')

  # Compute coherence scores ---------------------------------------------------------------------------------------------
  model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus,
                                                          texts=nested_tokens, start=start, limit=limit, step=step)

  # Plot coherence scores
  x = range(start, limit, step)
  plt.figure(figsize=(10,6), dpi=1200)
  plt.xlabel("Num Topics")
  plt.ylabel("Coherence score")
  plt.legend(("coherence_values"), loc='best')
  plt.xticks(x, rotation=90)
  plt.grid(True)
  plt.plot(x, coherence_values)
  plt.tight_layout()
  plt.savefig(workflow_folder + r'\visuals\lda_coherence.png')