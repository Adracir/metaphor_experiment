# TODO: visualize results from execute-experiment (with plotly?)
#   combine gutenberg and wiki data in one graphic
#   make separate graphics for word forms
#   make separate graphics for calculation methods
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TODO: does it make sense to have info from two files in one plot?
def output_to_plot(filename, filename2="", pos="all"):
    # fetch csv file by filename and save useful info in lists
    df = pd.read_csv(filename)
    pos_df = df[df['pos'] == pos]
    metaphor_names = pos_df['metaphor-name'].tolist()
    similarities = pos_df['mean_similarity'].tolist()
    baseline_performance = pos_df['baseline_performance'].tolist()
    test_statistic = pos_df['test_statistic'].tolist()
    p_value = pos_df['p_value'].tolist()
    # fetch csv file by second filename (if present) and save useful info in lists
    if filename2:
        df2 = pd.read_csv(filename2)
        pos_df2 = df2[df2['pos'] == pos]
        metaphor_names2 = pos_df2['metaphor-name'].tolist()
        similarities2 = pos_df2['mean_similarity'].tolist()
        baseline_performance2 = pos_df2['baseline_performance'].tolist()
        test_statistic2 = pos_df2['test_statistic'].tolist()
        p_value2 = pos_df2['p_value'].tolist()
        filename_parts2 = re.search('(.*?\/)(.*?)(\..*)', filename2)
        file_info2 = filename_parts2.group(2)
        splitted_infos2 = file_info2.split('_')
    # title
    filename_parts = re.search('(.*?\/)(.*?)(\..*)', filename)
    file_info = filename_parts.group(2)
    splitted_infos = file_info.split('_')
    # TODO: does this check make sense?
    if filename2 and (metaphor_names2 != metaphor_names or splitted_infos[4] != splitted_infos2[4] or
                      splitted_infos[6] != splitted_infos2[6]):
        print('!!! WARNING: Files are not equal in either used metaphors, used methods or weights !!! Please use only '
              'files that are equal in these aspects to plot a graph')
        return
    plt.title(f'Korpora: {splitted_infos[1]}{f" & {splitted_infos2[1]}" if filename2 else ""}, similarity measure: '
              f'{splitted_infos[4]}, POS: {pos}, weighted: {"yes" if splitted_infos[6] == "weighted" else "no"}')
    x = np.arange(len(similarities))
    ax = plt.gca()
    # TODO: maybe start with a little spacing on the left
    ax.set_xlim(0, len(metaphor_names))
    plt.xticks(x, metaphor_names, rotation ='vertical')
    plt.xlabel("Metaphors")
    plt.ylabel('Similarities')
    # plot similarities 1
    plt.plot(x, similarities, 'r-', label=f"Similarity between domains {splitted_infos[1]}")
    # plot baseline performance 1
    plt.plot(x, baseline_performance, 'b-', label=f"Similarity to random baseline {splitted_infos[1]}")
    if filename2:
        # plot similarities 2
        plt.plot(x, similarities2, 'm-', label=f"Similarity between domains {splitted_infos2[1]}")
        # plot baseline performance 2
        plt.plot(x, baseline_performance2, 'c-', label=f"Similarity to random baseline {splitted_infos2[1]}")
    '''x_filt = x[x > 3]
    x_filt_green = x[(p_value[x] < 0.01) & (test_statistic[x] >= 0)]
    y_filt_green = similarities[(p_value < 0.01) & (test_statistic >= 0)]
    plt.plot(x_filt_green, y_filt_green, 'gs', label="positively significant values")
    x_filt_red = x[(p_value < 0.01) & (test_statistic < 0)]
    y_filt_red = similarities[(p_value < 0.01) & (test_statistic < 0)]
    plt.plot(x_filt_red, y_filt_red, 'rs', label="negatively significant values")'''
    # TODO: add to legend
    # makes significance visible
    for i in range(len(x)):
        if p_value[i] < 0.01:
            color = 'g' if test_statistic[i] >= 0 else 'r'
            plt.plot(x[i], similarities[i], f'{color}s')
        if filename2 and p_value2[i] < 0.01:
            color2 = 'g' if test_statistic2[i] >= 0 else 'r'
            plt.plot(x[i], similarities2[i], f'{color2}s')
    plt.legend()
    # sets tight layout (so that nothing is cut out)
    plt.tight_layout()
    # save diagram
    fig = plt.gcf()
    fig.set_size_inches(10, 5.5)
    path = f'results/plots/{splitted_infos[1]}{f"-{splitted_infos2[1]}" if filename2 else ""}_{splitted_infos[4]}_{pos}_{"weighted" if splitted_infos[6] == "weighted" else ""}_plot.png'
    fig.savefig(path)
    plt.close(fig)
    print("plot saved")


output_to_plot('results/word2vec_gutenberg_1-8000u16001-26000_skipgram_cosine_all-ADJ-VERB-NOUN_weighted_results.csv',
               'results/word2vec_wiki_1-200000_skipgram-more-vocab2_cosine_all-ADJ-VERB-NOUN_weighted_results.csv')
