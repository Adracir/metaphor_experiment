# metaphor_experiment
a project for university, trying to explore common metaphors using word embeddings

## Requirements
All necessary requirements to run the code can be installed using 
```pip install -r requirements.txt```

## Usage
What can the code do and where?
- Clean text from wikipedia and gutenberg dumps: corpora.py
- generate Word2Vec embeddings using Gensim from the cleaned texts: embeddings.py
- evaluate the created embeddings over a set of human-annotation-based similarity scores: embeddings.py
- execute experiments for a given set of pairs of words that are thought to be associated to each other: main.py, also using calc.py
  - differentiating between:
    - corpus (gutenberg or wikipedia)
    - pos (ADJ, VERB, NOUN, or all)
    - similarity/distance-measure (cosine, canberra, euclidian or manhattan)
    - weights (double-weighting one word per metaphor per pos or not)
    - baseline (using existing sets of random vectors or generating new ones)
- save the experiment's results to csv files and plots: plot.py 

