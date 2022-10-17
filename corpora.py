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

    cleaned_texts = []
    for p_text in text:
        # TODO: think of sensible order of the different operations!
        # ignores whole text of page if it starts with #REDIRECT (means, page only redirects to other page)
        if not re.match('#REDIRECT', p_text):
            # removes &quot; (quotation mark)
            p_text = re.sub('&quot;', '', p_text)
            # removes stuff inside <ref> and </ref> (some references)
            p_text = re.sub('(<ref.*?>.*?</ref>|<ref.*?/>)', '', p_text)
            # removes stuff inside {||} (tables)
            p_text = re.sub('\{\|[\s\S]*?\|\}', '', p_text)  # TODO: generally replace .*? by [\s\S]*? ? would include newline chars
            # removes stuff inside {{}}(references, meta)
            p_text = re.sub('\{\{.*?\}\}', '', p_text)
            # accepts content left over from nested brackets, but cleans up the brackets themselves
            p_text = re.sub('\{\{|\}\}', '', p_text)
            # p_text = remove_text_inside_brackets(p_text, brackets="{}")
            # removes stuff inside <div> (html stylings) (also nested)
            # TODO: necessary? if so, do it!
            # removes '' (marks text as italic) and ''' (marks text as bold)
            p_text = re.sub('(\'){2,3}', '', p_text)
            # removes stuff inside [[Category: ]] (links to wiki categories)
            p_text = re.sub('\[\[Category\:.*?\]\]', '', p_text)
            # removes stuff inside [[File:]] including nested brackets (image metadata & image description)
            # p_text = remove_text_inside_brackets(p_text, ["\[\[File:", "\]\]"], ["\[\[", "\]\]"])
            # TODO: necessary? if so, do it!
            # removes remaining [[ ]] (links to other wiki page)
            # if two alternatives marked by |, use second (indicating visible text)
            p_text = remove_square_brackets_choose_alternative(p_text)
            cleaned_texts.append(p_text)
    return "".join(cleaned_texts)


# stuff will probably not be followed anymore, has no priority!
# It is very complex to remove content from brackets including nested ones
# TODO: improve code, also in loop-function?
def remove_text_inside_brackets(text, brackets, innerbrackets=[]):
    # for brackets containing of multiple characters, treated as regex
    if len(innerbrackets) > 0: # TODO: better (also) check brackets[0].len > 1?
        # split text at opening brackets
        text_list = re.split(brackets[0], text)
        print(f'text_list: {text_list}')
        # add first element to groups (before first opening bracket)
        groups = [text_list[0]]
        text_list.remove(text_list[0])
        # call function
        loop_text_list(text_list, groups, innerbrackets)
    else:
        # TODO: takes too long!! maybe adapt other case! only that innerbrackets equal brackets here... that makes it a bit more difficult
        groups = ['']
        # iterates through text, character by character
        for i in text:
            # if character indicates opening bracket, add new item to group (list)
            if i == brackets[0]:
                groups.append(i)
            # if character indicates closing bracket, and there is something contained in group,
            # remove last item from group
            elif i == brackets[1] and len(groups) > 1:
                groups.pop()
            # if it is any other character, add this to the last item of the group
            else:
                groups[-1] += i
            print(groups)
    # group string is returned
    return "".join(groups)


# TODO: improve code!
def loop_text_list(text_list, groups, innerbrackets):
    print(f'text_list: {text_list}, groups: {groups}')
    # after first opening bracket, iterate text_list
    for t in text_list:
        # if next innerbracket is opening bracket, append element to group list
        if (re.search(innerbrackets[0], t) and re.search(innerbrackets[1], t) and re.search(innerbrackets[0], t).regs[0][0] < re.search(innerbrackets[1], t).regs[0][0]) or (re.search(innerbrackets[0], t) and not re.search(innerbrackets[1], t)):
            print(f'!!!opening bracket in t: {t}!!!')
            inner_text_list = re.split(innerbrackets[0], t, 1)
            # add first part before [[ to groups
            groups.append(inner_text_list[0])
            # replace t by remaining stuff from t and call loop recursively
            text_list[0] = inner_text_list[1]
            loop_text_list(text_list, groups, innerbrackets)
        # if there is a match with another closing bracket, remove last element from group list
        elif (re.search(innerbrackets[0], t) and re.search(innerbrackets[1], t) and re.search(innerbrackets[1], t).regs[0][0] < re.search(innerbrackets[0], t).regs[0][0]) or (re.search(innerbrackets[1], t) and not re.search(innerbrackets[0], t)):  # TODO: what if case groups only contains one element?
            print(f'!!!closing bracket in t: {t}!!!')
            # split at first occurrence of closing tag
            inner_text_list = re.split(innerbrackets[1], t, 1)
            # remove last content of group if it contains more than one
            # add remaining part of t to last group item

            # add remaining part of t to text_list if groups has still more than 1 element
            if len(groups) > 1:
                text_list[0] = inner_text_list[1]
                groups.pop()
            else:
                text_list.remove(t)
                groups[-1] += inner_text_list[1]
            # recall function recursively
            loop_text_list(text_list, groups, innerbrackets)
        # else append text to last group list element
        else:
            groups[-1] += t
            print(f'!!!t added: {t}!!!')
        print(f'groups: {groups}')


# TODO: improve code
def remove_square_brackets_choose_alternative(text):
    text = re.sub("(\[\[.*?\|)(.*?)(\]\])", repl, text)
    text = re.sub("\[\[|\]\]", "", text)
    return text


def repl(matchobj):
    return matchobj.group(2)

# TODO:
#   maybe also remove
#       mathematical formulas
#       [ ] (links, but contains also other info)

