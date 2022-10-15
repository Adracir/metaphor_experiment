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
            # removes stuff inside {{}} (references, meta) &  inside {||} (tables), also if nested
            p_text = remove_text_inside_brackets(p_text, brackets="{}")
            # removes stuff inside <ref> and </ref> (some references)
            p_text = re.sub('(<ref.*?>.*?</ref>|<ref.*?/>)', '', p_text)
            # removes stuff inside <div> (html stylings) (also nested)

            # removes '' (marks text as italic) and ''' (marks text as bold)
            p_text = re.sub('(\'){2,3}', '', p_text)
            # removes stuff inside [[Category: ]] (links to wiki categories)
            p_text = re.sub('\[\[Category\:.*?\]\]', '', p_text)
            # TODO: remove stuff inside [[File:]] (image metadata & image description)
            #  and also take care of nested brackets! maybe rewrite remove_text_inside_brackets for it!
            cleaned_texts.append(p_text)

    print(f'text length after: {len(cleaned_texts)}')
    print(f'cleaned texts: {cleaned_texts}')
    return ""


def remove_text_inside_brackets(text, brackets):
    # for brackets containing of multiple characters, treated as regex
    if len(brackets[0]) > 1:
        groups = ['']
        for m in re.finditer(brackets[0]):
            if i == brackets[0]:
                groups.append(i)
            elif i == brackets[1] and len(groups) > 1:
                groups.pop()
            else:
                groups[-1] += i
        return "".join(groups)
    else:
        groups = ['']
        # iterates through text, character by character
        for i in text:
            # if character indicates opening bracket, add to group
            if i == brackets[0]:
                groups.append(i)
            # if character indicates closing bracket, and there is something contained in group,
            # remove last item from group
            elif i == brackets[1] and len(groups) > 1:
                groups.pop()
            # if it is any other character, add this to the end of the group
            else:
                groups[-1] += i
        return "".join(groups)


    # TODO:
    #   remove
    #       maybe mathematical formulas
    #       [ ] (links, but contains also other info)
    #       [[ ]] (link to other wiki page)
    #           if two alternatives marked by |, use second (indicating visible text)

preprocess_wiki_dump()
