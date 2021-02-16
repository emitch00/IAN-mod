import os
import re

input_fname = 'devset/tweets.txt
output_fname = 'data/post_train.txt

#need to fix
a_name = open("tweets.txt", "r")
tweets = a_name.readlines()
a_fname.close()

#delete first line
del tweets[1]

#need to find fake/real
for i in tweets:
    if(tweets[i].find('	fake') != -1)
         news[i] = 'fake'
    else
         news[i] = 'real'
#keep only userID and tweet
tweets = [i.split('http', 1)[0] for i in tweets]
#remove all numeric values and whitespace after
tweets = [re.sub("[^a-zA-Z ]+", "", i) for i in tweets]
#tweets = [''.join(filter(i.isalpha(), tweets)) for i in tweets]


#search for aspect terms
output_fname = open("post_train.txt", "w")
for i in tweets:
    while(tweets[i].find(aspect_term) != -1){
        #replace aspect terms with aspect_term
        re.sub("aspect_term", "aspect_term")
        #to output file

        #line
        #aspect term
        #fake/real
        output = [tweet[i] + "\n", aspect_term + "\n", news[i] + "\n"]
        output_fname.writelines(output)
    }






#import os
#import unicodedata
#import xml.etree.ElementTree as ET
#from errno import ENOENT


#input_fname = 'data/restaurant/train.xml'
#output_fname = 'data/restaurant/train.txt'

#if not os.path.isfile(input_fname):
    #raise IOError(ENOENT, 'Not an input file', input_fname)
    
#with open(output_fname, 'w') as f:
    #tree = ET.parse(input_fname)
    #root = tree.getroot()
    #sentence_num = 0
    #aspect_num = 0
    #for sentence in root.iter('sentence'):
        #sentence_num = sentence_num + 1
        #text = sentence.find('text').text
        #for asp_terms in sentence.iter('aspectTerms'):
            #for asp_term in asp_terms.findall('aspectTerm'):
                #if asp_term.get('polarity') != 'conflict' and asp_term.get('target') != 'NULL':
                    #aspect_num = aspect_num + 1
                    #new_text = ''.join((text[:int(asp_term.get('from'))], 'aspect_term', text[int(asp_term.get('to')):]))
                    #f.write('%s\n' % new_text.strip())
                    #f.write('%s\n' % asp_term.get('target'))
                    #f.write('%s\n' % asp_term.get('polarity'))
                    #print("Read %s sentences %s aspects" % (sentence_num, aspect_num))
                    
