import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def output_to_plot(filename, pos="all"):
    """
    create and save a scattered plot to results/plots from result data from the experiment
    :param filename: path to the result csv file
    :param pos: only one part-of-speech can be shown in one graph, so choose from "ADJ", "all", "VERB" and "NOUN"
    """
    # fetch csv file by filename and save useful info in lists
    df = pd.read_csv(filename)
    pos_df = df[df['pos'] == pos]
    metaphor_names = pos_df['metaphor_name'].tolist()
    similarities = pos_df['mean_similarity'].tolist()
    baseline_performance = pos_df['baseline_performance'].tolist()
    test_statistic = pos_df['test_statistic'].tolist()
    p_value = pos_df['p_value'].tolist()
    # title
    # TODO: test if works with nested directories, too
    filename_parts = re.search('(.*?\/)(.*?)(\..*)', filename)
    file_info = filename_parts.group(2)
    # TODO: test if works like this, test if can be one liner
    split_prefix = file_info.split('-')
    prefix = split_prefix[0]
    splitted_infos = file_info.split('_')
    plt.title(f'Korpus: {splitted_infos[1]}, distance measure: '
              f'{splitted_infos[4]}, POS: {pos}, {splitted_infos[6]}, Baseline: {prefix}')
    # prepare axes
    x = np.arange(len(similarities))
    ax = plt.gca()
    # TODO: maybe start with a little spacing on the left
    ax.set_xlim(0, len(metaphor_names))
    plt.xticks(x, metaphor_names, rotation ='vertical')
    plt.xlabel("Metaphors")
    plt.ylabel('Similarities')
    # plot similarities
    plt.plot(x, similarities, 'r-', label=f"Similarity between domains {splitted_infos[1]}")
    # plot baseline performance
    plt.plot(x, baseline_performance, 'b-', label=f"Similarity to random baseline {splitted_infos[1]}")
    # TODO: make less verbose?
    # make significance visible
    negative_lgd_label = False
    positive_lgd_label = False
    for i in range(len(x)):
        if p_value[i] < 0.01:
            color = 'g' if test_statistic[i] >= 0 else 'r'
            lgd_label = ''
            if test_statistic[i] >= 0 and not positive_lgd_label:
                lgd_label = 'significant (positive)'
                positive_lgd_label = True
            if test_statistic[i] < 0 and not negative_lgd_label:
                lgd_label = 'significant (negative)'
                negative_lgd_label = True
            plt.plot(x[i], similarities[i], f'{color}s', label=lgd_label)
    plt.legend()
    # set tight layout (so that nothing is cut out)
    plt.tight_layout()
    # save diagram
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    path = f'results/plots/{prefix}-{splitted_infos[1]}_{splitted_infos[4]}_{pos}_{splitted_infos[6]}_plot.png'
    fig.savefig(path)
    plt.close(fig)
    print("plot saved")
