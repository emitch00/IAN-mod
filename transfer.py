import nltk
ntlk.download('punkt')
nltk.download('stopwords')

import re
from nltk.tokenize import word_tokenize

file = open("/content/groundtruth.txt", "r")
#commonWordsFile = open("/content/commonWords.txt", "r")

count = {}

for line in file:
  aspect_term_list = open("/content/sortedGroundtruth.txt", "r")
  word_token = re.split(" |\t", line)
  fact = word_token[len(word_token) - 1]
  fact = fact.rstrip("\n")
  # remove extra information about tweet, like date, time
  for i in range (0, 7):
    word_token.pop()

  #print(word_token)
  # make sure only processing the lines with real data -
  if fact == 'real' or fact == 'fake':
    for i in word_token: 
      i = i.lower()
      # filter out empty strings, numbers, http links, @handles, and 'rt'
      if len(i) > 0 and (not (i.isdigit())) and i.find('http') == -1 and i[0] != "@" and i != 'rt':
          # check if word is an aspect term
          for word in aspect_term_list:
            word = word.rstrip("\n")
            if word == i:
              if i not in count: 
                count[i] = 0 
              count[i]+=1
              i = 'aspect_term'
            
file.close()
commonWordsFile.close()

maplist = {}
for i in maplist:
  maplist[i] = count.get(i)
  print(maplist)

#new_file = open("/content/data_sort.txt", "w")
#for key, value in sorted_count.items():
  #new_file.write(line +" : "+str(value)+"\n")

#new_file.close()
