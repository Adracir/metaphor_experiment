# metaphor_experiment

a university project, trying to explore common metaphors using word embeddings

## Requirements

All necessary requirements to run the code can be installed using 
```pip install -r requirements.txt```

## Usage

### What can the code do?

- Clean text from wikipedia and gutenberg dumps: ``corpora.py``
- generate Word2Vec embeddings using Gensim from the cleaned texts: ``embeddings.py``
- evaluate the created embeddings over a set of human-annotation-based similarity scores: ``embeddings.py``
- execute experiments for a given set of pairs of words that are thought to be associated to each other: ``main.py``, also using ``calc.py``
  - differentiating between:
    - corpus (gutenberg or wikipedia)
    - Part-of-Speech (POS) (ADJ, VERB, NOUN, or all)
    - similarity/distance-measure (cosine, canberra, euclidian or manhattan)
    - weights (double-weighting one word per metaphor per POS or not)
    - baseline (using existing sets of random vectors or generating new ones)
  - also allows evaluation on whether some words needed for the experiment are not in the vocabulary of the word embeddings. 
  Ideally, these should be fixed to obtain good results, though.
- save the experiment's results to csv files and combine the information from these files to form a useful basis to analyze 
the data concerning the different parameters: ``main.py``
- generate plots to visualize parts of the results: ``plot.py``

### What resources are (not) available in this repository?

Some resources that are needed to execute the code had to be omitted in this repository as they exceed the file size limit
of GitHub. The following list clarifies the project structure and mentions omitted files
- ``data``:
  - contains ``human_relatedness.csv``, based on Rubenstein and Goodenough (1965), used for embedding evaluation
  - contains ``word_sets.csv``, the words belonging to different metaphors, divided in two domains per metaphor, used for experiment
  - contains ``random_vector_sets.npy`` for both corpora, used as baseline (*savedBL*)
  - subfolder ``gutenberg`` 
    - contains small sample of cleaned text (``cleaned_texts_from_1_to_25.txt``). The cleaned text files actually used 
    in the experiment are omitted here
    - contains a list of english and american novelists (``english_american_authors.txt``) used to filter the gutenberg index files
    - contains the filtered list of indices (``indices.txt``) of gutenberg texts used in the experiment and the corresponding lookup file with more information for each index (``indices_lookup.txt``)
    - gutenberg index files and the raw texts are omitted
  - subfolder ``wiki`` 
    - contains small sample of cleaned text (``cleaned_texts_from_1_to_100.txt``). The cleaned text files actually used 
    in the experiment are omitted here
    - the raw wikipedia dump (xml format) is omitted
- ``models``:
  - trained KeyedVectors and the whole folder are omitted
- ``results``:
  - contains all CSV files with the results for the executed experiments with all parameter settings
  - subfolder ``confront_files`` contains condensed data from the results for the analysis of each parameter
  - subfolder ``plots`` contains diagrams visualizing the results

## Context

This work has been done for the examination in the module *Verarbeitung von Textdaten* in the master course *Informationsverarbeitung*
at the Institute for Digital Humanities at University of Cologne. Further information about the underlying assumptions and
questions, the theoretical background and the interpretation of the results can be found in a written document in german 
language passed over in the context of the examination.

