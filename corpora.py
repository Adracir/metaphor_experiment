import re
from xml.etree import ElementTree as ET
import time
import os
from os.path import exists

import utils

root = 'C:/Users/adrac/Documents/Uni/Embeddings/metaphor_experiment'
WIKI_DUMP = "/data/wiki/enwiki-latest-pages-articles-multistream.xml"
namespaces = {'wiki_ns': 'http://www.mediawiki.org/xml/export-0.10/'}
AUTHOR_LIST = "data/gutenberg/english_american_authors.txt"


def preprocess_wiki_dump(begin_at, end_at):
    """
    clean wikipedia text at a given chunk and save cleaned text to txt file
    :param begin_at: begins at this page (inclusive)
    :param end_at: ends at this page (inclusive)
    :return: cleaned text of the specified pages
    """
    start = time.time()
    cleaned_texts = []
    counter = 0
    # parse tree and take only text elements
    for event, elem in ET.iterparse(root + WIKI_DUMP, events=("start", "end")):
        if event == "end":
            if elem.tag == '{' + namespaces.get('wiki_ns') + '}text':
                counter += 1
                if counter < begin_at:
                    continue
                elif begin_at <= counter <= end_at:
                    end = time.time()
                    print(f'time taken before start: {end-start} seconds')
                    # take only inner text
                    p_text = ''.join(elem.itertext())
                    p_text = clean_wiki_text_from_one_page(p_text, counter)
                    cleaned_texts.append(p_text)
                else:
                    print(f'stopped at text {counter}')
                    break
    # save cleaned text
    filename = f'cleaned_texts_from_{begin_at}_to_{end_at}.txt'
    filepath = root + '/data/wiki/' + filename
    cleaned_text_string = "".join(cleaned_texts)
    utils.save_string_to_txt_file(filepath, cleaned_text_string)
    print(f'saved {len(cleaned_texts)} cleaned texts to file {filename}')
    return cleaned_text_string


def clean_wiki_text_from_one_page(p_text, counter):
    """
    clean the text from a wikipedia page from metadata, so that pure text remains
    :param p_text: page's text as read from dump
    :param counter: number indicating wiki page
    :return: the cleaned text
    """
    # ignores whole page if it starts with #REDIRECT or {{wiktionary (only redirects to other pages)
    if not re.match(r'(#REDIRECT)|(\{\{wiktionary)|(\[\[Wikipedia:Free_On-line_Dictionary_of_Computing/symbols)',
                    p_text):
        # removes &quot; (quotation mark)
        p_text = re.sub('&quot;', '', p_text)
        # removes everything inside {||} (tables)
        p_text = re.sub(r'\{\|[\s\S]*?\|\}', '', p_text)
        # removes everything inside {{}}(references, meta)
        p_text = re.sub(r'\{\{[\s\S]*?\}\}', '', p_text)
        # accepts content left over from nested brackets, but cleans up the brackets themselves
        p_text = re.sub(r'\{\{|\}\}', '', p_text)
        # removes '' (marks text as italic) and ''' '''(marks text as bold)
        p_text = re.sub(r'(\'){2,3}', '', p_text)
        # removes everything inside [[Category: ]] (links to wiki categories)
        p_text = re.sub(r'\[\[Category\:.*?\]\]', '', p_text)
        # removes everything inside [[File:]] (image metadata & image description)
        p_text = re.sub(r'\[\[File:.*?\|+.*(\]\])+', '', p_text)
        # removes remaining square brackets and chooses what content to keep
        p_text = re.sub(r'(\[\[.[^\[\]]*?\|)(.*?)(\]\])', repl, p_text)
        # clean up all [[ ]] that are left
        p_text = re.sub(r'\[\[|\]\]', '', p_text)
        # removes leftovers from tables
        p_text = re.sub(r'\|.[^ \n]*', '', p_text)
        # clean up also links in [ ]
        p_text = re.sub(r'\[.*?\]', '', p_text)
        # br-tags (makes next step less complex, especially for long lists
        p_text = re.sub('<br>', ' ', p_text)
        # removes remaining html tags and their content
        p_text = re.sub(r'(<\w+?.[^\/]*?>[\s\S]*?<\/\w+?>)|(<\w+?.*?\/>)', '', p_text)
        # removes html comments
        p_text = re.sub('<!--.*?-->', '', p_text)
        # removes stars
        p_text = re.sub(r'\*', '', p_text)
        # removes newlines
        p_text = re.sub('\n', ' ', p_text)
        print(f'text {counter} cleaned')
        return p_text
    else:
        print(f'text {counter} ignored')
    return ''


def repl(matchobj):
    """
    replace regex match object with its second match group
    :param matchobj: regex match object containing at least 2 groups
    :return: second match group
    """
    return matchobj.group(2)


def parse_authors_list():
    """
    read author list file and convert to python list
    :return: python list containing author names from file as separate elements
    """
    with open(AUTHOR_LIST, 'r', encoding='utf8') as file:
        author_list = file.readlines()
        for i in (range(len(author_list))):
            author_cleaned = re.sub(r'(\s\(.*)?(,.*)?\n?', '', author_list[i])
            author_list[i] = author_cleaned
    return author_list


def extract_useful_indices_from_gutenberg_index_files(author_list):
    """
    iterate gutenberg index files to extract the files that are useful
    (only texts written in english by authors from author list)
    1. save indices of useful files in indices.txt
    2. save more info about each useful file in indices_lookup.txt
    :param author_list: list of author names, can be generated by parse_authors_list
    :return: list of indices
    """
    indices = []
    # assign directory
    directory = 'data/gutenberg'
    with open('data/gutenberg/indices_lookup.txt', 'w', encoding='utf8') as i_f:
        i_f.write('----- START -----\n')
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a gutenberg index file
        if os.path.isfile(f) and re.match('GUTINDEX.*', filename):
            print(f'filename to be opened: {filename}')
            # open file and iterate lines
            with open(f, 'r', encoding='utf8') as file:
                lines = file.readlines()
                last_read_relevant_title = ''
                last_read_title = ''
                for line in lines:
                    # extract info about line
                    possible_author_match = re.search(r'(.*?)(by)( .*?)(\d{1,5}\w?\n)', line)
                    title_match = re.match(r'.*?\d{1,5}\w?\n', line)
                    lang_match = re.search(r'(\[Language:) (.*?)(\])', line)
                    # check if author in author-list
                    author_name = [ele for ele in author_list if(ele in line)]
                    if title_match:
                        last_read_title = line
                    if possible_author_match and bool(author_name):
                        title = possible_author_match.group(1)
                        # ignore audio files and append match to lookup file
                        if 'Audio:' not in title:
                            last_read_relevant_title = line
                            indices.append(possible_author_match.group(4))
                            # writes info to lookup file
                            with open('data/gutenberg/indices_lookup.txt', 'a', encoding='utf8') as i_f:
                                i_f.write(f'FILE:   {filename}\n')
                                i_f.write(f'LINE:   {line}')
                                i_f.write(f'TITLE:  {title}\n')
                                i_f.write(f'AUTHOR: {author_name}\n')
                                i_f.write(f'INDEX:  {possible_author_match.group(4)}-----\n')
                    # if language is explicitly mentioned and not english, remove corresponding title from list
                    elif lang_match:
                        lang = re.sub(r'\s+', '', lang_match.group(2))
                        if (lang != 'English') and (last_read_title == last_read_relevant_title):
                            indices.pop()
    # write all generated indices to file
    with open('data/gutenberg/indices.txt', 'w') as i_f:
        for index in indices:
            i_f.write(f"{index}")
    return indices


def preprocess_gutenberg_dump(begin_at, end_at):
    """
    clean a range of gutenberg texts of meta information, footnotes and other elements not useful for embedding
    creation and save cleaned text to file
    :param begin_at: index to start at (must not be a real index, inclusive)
    :param end_at: index to end at (must not be a real index, exclusive)
    :return: cleaned string, combining all files in the given range
    """
    raw_files = 'data/gutenberg/raw_files'
    existing = []
    cleaned_texts = []
    # iterates all gutenberg txt files in the given range of possible indices
    for i in range(begin_at, end_at):
        path = generate_path_to_gutenberg_file(i)
        if exists(raw_files + path):
            existing.append(i)
            print(i)
            # if file exists at the path, start cleaning text
            with open(raw_files + path, 'r') as file:
                uncleaned_str = file.read()
                cleaned = clean_gutenberg_text_from_one_file(uncleaned_str)
                cleaned_texts.append(cleaned)
    # save cleaned text
    filename = f'cleaned_texts_from_{begin_at}_to_{end_at}.txt'
    filepath = root + '/data/gutenberg/' + filename
    cleaned_text_string = "".join(cleaned_texts)
    utils.save_string_to_txt_file(filepath, cleaned_text_string)
    print(f'saved {len(cleaned_texts)} cleaned texts to file {filename}')
    return cleaned_text_string


def generate_path_to_gutenberg_file(i):
    """
    generate a path to a gutenberg file, based on the number of the file
    example: number 11 would be found in /1/11/11.txt, 1100 in /1/1/0/1100/1100.txt etc.
    :param i: number of the file named [number].txt
    :return: complete path to the file relative to the root of the gutenberg corpus
    """
    x = [int(a) for a in str(i)]
    # generate path to look for file
    path = ''
    if len(x) == 1:
        path = f'/{i}/{i}.txt'
    else:
        for num in x[:-1]:
            path += f'/{num}'
        path += f'/{i}/{i}.txt'
    return path


def clean_gutenberg_text_from_one_file(f_text):
    """
    clean the text from a gutenberg file from metadata, so that pure text remains
    :param f_text: the text as read from the file
    :return: cleaned text
    """
    # removing beginning and end meta info
    split_beginning = re.split(r'\*\*\*\s?START OF.*?\n?.*?\*\*\*', f_text)
    without_beginning = split_beginning[1] if len(split_beginning) > 1 else f_text
    split_end = re.split(r'\*\*\*\s?END OF.*?\n?.*?\*\*\*', without_beginning)
    without_end = split_end[0] if len(split_end) > 1 else without_beginning
    # removing footnotes and underscores
    cleaned = re.sub(r'\[.*?\]', '', without_end)
    cleaned = re.sub('_', '', cleaned)
    # removing some remaining meta info
    cleaned = re.sub('End of the Project Gutenberg EBook.*?\n', '', cleaned)
    cleaned = re.sub('Produced by .*?HTML version by.*?\n', '', cleaned)
    return cleaned
