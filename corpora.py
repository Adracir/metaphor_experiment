import re
from xml.etree import ElementTree as ET

# TODO: preprocess Wikipedia
#  (https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67)

root = 'C:/Users/adrac/Documents/Uni/Embeddings/metaphor_experiment'
# WIKI_DUMP = "/data/wiki/enwiki-latest-pages-articles-multistream.xml"
WIKI_DUMP = "/data/wiki/small_test_wiki.xml"
namespaces = {'wiki_ns': 'http://www.mediawiki.org/xml/export-0.10/'}


def preprocess_wiki_dump():
    tree = ET.parse(root + WIKI_DUMP)
    # get only <page><text>
    text_elements = tree.findall('.//wiki_ns:page//wiki_ns:text', namespaces)
    # TODO: find better method to parse text than to use ''.join(e.itertext())
    # contains list of text structured by pages
    text = [''.join(e.itertext()) for e in text_elements]

    print(f'text length before: {len(text)}')
    cleaned_texts = []
    for p_text in text:
        # TODO: think of sensible order of the different operations!
        # ignores whole text of page if it starts with #REDIRECT (means, page only redirects to other page)
        if not re.match('#REDIRECT', p_text):
            # removes &quot; (quotation mark)
            p_text = re.sub('&quot;', '', p_text)
            # removes stuff inside {{}} (references, meta)
            # TODO: solve problem that 18 {{ remain because they were nested
            p_text = re.sub('{{.[^{{]*}}', '', p_text)
            cleaned_texts.append(p_text)

    print(f'text length after: {len(cleaned_texts)}')
    print(f'cleaned texts: {cleaned_texts}')

    # TODO:
    #   remove
    #       '' (marks text as italic)
    #       ''' (marks text as bold)
    #       stuff inside {| (tables)
    #       stuff inside &lt;ref and &gt; (some references, if all starting from &lt;, also mathematical formulas etc
    #        might be excluded)
    #       [[ ]] (link to other wiki page)
    #           if two alternatives marked by |, use second (indicating visible text)
    #           remove whole content inside those brackets if it begins with "File:" (image metadata & image
    #           description)
    return ""

preprocess_wiki_dump()
