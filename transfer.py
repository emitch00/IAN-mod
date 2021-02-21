import nltk
ntlk.download('punkt')
nltk.download('stopwords')

import re
from nltk.tokenize import word_tokenize

file = open("/content/groundtruth.txt", "r")
#commonWordsFile = open("/content/commonWords.txt", "r")

count = {}

for line in file:
  commonWordsFile = open("/content/commonWords.txt", "r")
  english = True
  unique = True
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
        # filter out words with non english letters
        for letter in i:
          if ord(letter) < 97 or ord(letter) > 122:
            english = False
            break
        # only add to map if word has english letters
        if (english):
          # check that word isnt a common one
          commonWordsFile = open("/content/commonWords.txt", "r")
          for word in commonWordsFile:
            word = word.rstrip("\n")
            if word == i:
              unique = False
              break

          if unique:
            i = i + ", " + fact
            if i not in count: 
              count[i] = 0 
            count[i]+=1
            
file.close()
commonWordsFile.close()

sorted_list = sorted(count, reverse = True, key = lambda x: count[x])
sorted_count = {}
for i in sorted_list:
  sorted_count[i] = count.get(i)

new_file = open("/content/sortedGroundtruth.txt", "w")
for key, value in sorted_count.items():
  new_file.write(key+" : "+str(value)+"\n")

new_file.close()
