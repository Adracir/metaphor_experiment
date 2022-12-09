# TODO: visualize results from execute-experiment (with plotly?)
#   combine gutenberg and wiki data in one graphic
#   make separate graphics for word forms
#   make separate graphics for calculation methods

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def output_to_plot(filename, pos="all"):
    # TODO: adjust to goals:
    #   - get info from different files (= diff corpora)
    # fetch csv file by filename and save useful info in lists
    df = pd.read_csv(filename)
    pos_df = df[df['pos'] == pos]
    metaphor_names = pos_df['metaphor-name'].tolist()
    similarities = pos_df['similarity'].tolist()
    baseline_performance = pos_df['baseline_performance'].tolist()
    # title
    # TODO: adjust title
    plt.title(f'Korpora: googlenews, method: cosine, POS: {pos}, weighted: no')
    x = np.arange(len(similarities))
    ax = plt.gca()
    # TODO: maybe start with a little spacing on the left
    ax.set_xlim(0, len(metaphor_names))
    plt.xticks(x, metaphor_names, rotation ='vertical')
    plt.xlabel("Metaphors")
    plt.ylabel('Similarities')
    # TODO: maybe make p-value visible if it exists:
    #  as differently formed points for significant vs not significant values
    # plot similarities
    plt.plot(x, similarities, 'go')
    plt.plot(x, similarities, 'r-', label="Similarity between domains")
    # plot baseline performance
    plt.plot(x, baseline_performance, 'go')
    plt.plot(x, baseline_performance, 'b-', label="Similarity to random baseline")
    plt.legend()
    # sets tight layout (so that nothing is cut out)
    plt.tight_layout()
    # save diagram
    fig = plt.gcf()
    fig.set_size_inches(10, 5.5)
    # TODO: generate path depending on input
    path = 'results/plots/diagramm.png'
    fig.savefig(path)
    plt.close(fig)
    print("plot saved")


output_to_plot('results/googlenews_mean_vector_results.csv')
