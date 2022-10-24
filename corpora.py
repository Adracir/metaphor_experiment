import re
from xml.etree import ElementTree as ET
import time

root = 'C:/Users/adrac/Documents/Uni/Embeddings/metaphor_experiment'
WIKI_DUMP = "/data/wiki/enwiki-latest-pages-articles-multistream.xml"
WIKI_DUMP_SMALL = "/data/wiki/small_test_wiki.xml"
namespaces = {'wiki_ns': 'http://www.mediawiki.org/xml/export-0.10/'}


# TODO: enable cleaning chunks and saving to differently named files, making it possible to stop and continue cleaning
def preprocess_wiki_dump_timed_cached_test(begin_at, end_at):
    """
    cleans wikipedia text at a wanted chunk and saves cleaned text to txt file
    :param begin_at: begins at this page (inclusive)
    :param end_at: ends at this page (inclusive)
    :return: cleaned text of the taken pages
    """
    start = time.time()
    cleaned_texts = []

    # counts found elements to see if some stuff is useless:
    '''quotations = 0
    tables = 0
    references = 0
    curved_brackets = 0
    italic_bold_marks = 0
    categories = 0
    files = 0
    square_brackets_content = 0
    double_square_brackets = 0
    table_leftovers = 0
    links = 0
    html_tags = 0
    html_comments = 0
    stars = 0'''

    counter = 0
    # parse tree and take only text elements
    for event, elem in ET.iterparse(root + WIKI_DUMP, events=("start", "end")):
        if event == "end":
            if elem.tag == '{' + namespaces.get('wiki_ns') + '}text':
                counter += 1
                if counter < begin_at:
                    pass
                elif begin_at <= counter <= end_at:
                    end = time.time()
                    print(f'time taken before start: {end-start} seconds')
                    # TODO: maybe find better method to parse text than to use ''.join(p_text.itertext())
                    # take only inner text
                    p_text = ''.join(elem.itertext())
                    # writes uncleaned text to file, needed for analysis of long-cleaning texts
                    # uncleaned_file = open(f'{root}/data/wiki/uncleaned_{counter}.txt', 'w', encoding='utf-8')
                    # uncleaned_file.write((p_text))
                    # uncleaned_file.close()


                    # TODO: maybe exclude texts starting with {{Commons cat
                    #  as they seem to take long to process... but may contain also useful content :/
                    #   or maybe skip some known long cleaning texts by counter, e.g. 527114
                    # ignores whole page if it starts with #REDIRECT or {{wiktionary (only redirects to other pages)
                    if not re.match('(#REDIRECT)|(\{\{wiktionary)|(\[\[Wikipedia:Free_On-line_Dictionary_of_Computing/symbols)', p_text):
                        # removes &quot; (quotation mark)
                        # quotations += len(re.findall("&quot;", p_text))
                        p_text = re.sub('&quot;', '', p_text)
                        # removes stuff inside {||} (tables)
                        # tables += len(re.findall("\{\|[\s\S]*?\|\}", p_text))
                        p_text = re.sub('\{\|[\s\S]*?\|\}', '', p_text)
                        # removes stuff inside {{}}(references, meta)
                        # references += len(re.findall('\{\{[\s\S]*?\}\}', p_text))
                        p_text = re.sub('\{\{[\s\S]*?\}\}', '', p_text)
                        # accepts content left over from nested brackets, but cleans up the brackets themselves
                        # curved_brackets += len(re.findall('\{\{|\}\}', p_text))
                        p_text = re.sub('\{\{|\}\}', '', p_text)
                        # removes '' (marks text as italic) and ''' '''(marks text as bold)
                        # italic_bold_marks += len(re.findall('(\'){2,3}', p_text))
                        p_text = re.sub('(\'){2,3}', '', p_text)
                        # removes stuff inside [[Category: ]] (links to wiki categories)
                        # categories += len(re.findall('\[\[Category\:.*?\]\]', p_text))
                        p_text = re.sub('\[\[Category\:.*?\]\]', '', p_text)
                        # removes stuff inside [[File:]] (image metadata & image description)
                        # files += len(re.findall('\[\[File:.*?\|+.*(\]\])+', p_text))
                        p_text = re.sub('\[\[File:.*?\|+.*(\]\])+', '', p_text)
                        # removes remaining square brackets and chooses what content to keep
                        # square_brackets_content += len(re.findall("(\[\[.[^\[\]]*?\|)(.*?)(\]\])", p_text))
                        p_text = re.sub("(\[\[.[^\[\]]*?\|)(.*?)(\]\])", repl, p_text)
                        # clean up all [[ ]] that are left
                        # double_square_brackets += len(re.findall("\[\[|\]\]", p_text))
                        p_text = re.sub("\[\[|\]\]", "", p_text)
                        # removes leftovers from tables
                        # table_leftovers += len(re.findall('\|.[^ \n]*', p_text))
                        p_text = re.sub('\|.[^ \n]*', '', p_text)
                        # clean up also links in [ ]
                        # links += len(re.findall('\[.*?\]', p_text))
                        p_text = re.sub('\[.*?\]', '', p_text)
                        # TODO: too complex? decomplexify? :D
                        # removes remaining html tags and their content
                        # html_tags += len(re.findall('(<\w+?.[^\/]*?>[\s\S]*?<\/\w+?>)|(<\w+?.*?\/>)', p_text))
                        p_text = re.sub('(<\w+?.[^\/]*?>[\s\S]*?<\/\w+?>)|(<\w+?.*?\/>)', '', p_text)
                        # removes html comments
                        # html_comments += len(re.findall('<!--.*?-->', p_text))
                        p_text = re.sub('<!--.*?-->', '', p_text)
                        # removes stars
                        # stars += len(re.findall('\*', p_text))
                        p_text = re.sub('\*', '', p_text)
                        # removes newlines
                        p_text = re.sub('\n', ' ', p_text)
                        cleaned_texts.append(p_text)
                        print(f'text {counter} cleaned')
                    else:
                        print(f'text {counter} ignored')
                else:
                    print(f'stopped at text {counter}')
                    break
    # save cleaned text
    file_name = f'cleaned_texts_from_{begin_at}_to_{end_at}.txt'
    text_file = open(root + '/data/wiki/' + file_name, 'w', encoding="utf-8")
    cleaned_text_string = "".join(cleaned_texts)
    text_file.write(cleaned_text_string)
    text_file.close()
    print(f'saved {len(cleaned_texts)} cleaned texts to file {file_name}')
    '''print(f'quotations = {quotations}')
    print(f'tables = {tables}')
    print(f'references = {references}')
    print(f'curved_brackets = {curved_brackets}')
    print(f'italic_bold_marks = {italic_bold_marks}')
    print(f'categories = {categories}')
    print(f'files = {files}')
    print(f'square_brackets_content = {square_brackets_content}')
    print(f'double_square_brackets = {double_square_brackets}')
    print(f'table_leftovers = {table_leftovers}')
    print(f'links = {links}')
    print(f'html_tags = {html_tags}')
    print(f'html_comments = {html_comments}')'''
    return cleaned_text_string


def repl(matchobj):
    return matchobj.group(2)

# TODO: problem: the later you start, the more "foreplay" there is (e.g. 108s before 300000)
# TODO: maybe make automated way of cleaning chunks of the (whole?) text
start = time.time()
preprocess_wiki_dump_timed_cached_test(begin_at=527114, end_at=527114)
end = time.time()
print(f'time taken in seconds: {end - start}')
