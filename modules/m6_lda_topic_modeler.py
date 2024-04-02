from m1_main import *

import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def lda_modeler(df, topic_num, topn_topics):

  nested_tokens = [list(token_list) for token_list in df['Composite_tokens']]
  dictionary = Dictionary(nested_tokens)
  corpus = [dictionary.doc2bow(token_list) for token_list in nested_tokens]
  
  # Running LDA on the mini-corpus
  lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num, random_state=100,
                        update_every=0, passes=10, iterations=10, per_word_topics=True, minimum_probability=0.25)
  
  results = []

  # Extract the main topic for the current document
  for bow in corpus:
    document_topics = lda_model.get_document_topics(bow)
    
    if document_topics:
      dominant_topic = max(document_topics, key=lambda x: x[1])[0]
      topic_prob = max(document_topics, key=lambda x: x[1])[1]
      topic_prob = f'{topic_prob: .2%}'
      topic_keywords = [word for word, prob in lda_model.show_topic(dominant_topic, topn=topn_topics)]
      topic_keywords_with_prob = [prob for word, prob in lda_model.show_topic(dominant_topic, topn=topn_topics)]
      # ", ".join([f'{prob:.2%} {word}' for word, prob in lda_preprocessor_model.show_topic(dominant_topic, topn=topn_topics)])
    else:
      dominant_topic = None
      topic_prob = "0%"
      topic_keywords = []
      topic_keywords_with_prob = "No dominant topic"

    results.append({
      'Topic_Num': dominant_topic,
      'Topic_Prob': str(topic_prob),
      'Topic_Keywords_Prob': topic_keywords_with_prob,
      'Topic_Keywords': topic_keywords
    })

  return lda_model, corpus, dictionary, results

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
  
df = pd.read_pickle(lda_preprocessed_df_path)

lda_model, corpus, dictionary, results = lda_modeler(df, 25, 10)

results_df = pd.DataFrame(results)

# Validation step to check for empty lists in 'Topic_Keywords'
if any(results_df['Topic_Keywords'].apply(lambda x: x == [])):
    print(f"Warning: Empty lists found in 'Keywords'")

merged_df = pd.merge(df, results_df, left_index=True, right_index=True)
# merged_df = pd.concat([df, results_df], axis=1) # Same result as above

merged_df.to_excel(workflow_folder + r'\excel\lda_topics.xlsx', index=False)
merged_df.to_pickle(workflow_folder + r'\pickle\lda_topics.pickle')

# Check topic separation ------------------------------------------------------------------------------------------------
vis_data = gensimvis.prepare(lda_model, corpus, dictionary=dictionary)
pyLDAvis.save_html(vis_data, workflow_folder + r'\visuals\lda_pyldavis.html')