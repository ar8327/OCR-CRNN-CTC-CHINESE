import os
dictionary_dir = "../dictionaries/"

word_count = []
words = []
for file in os.listdir(dictionary_dir):
    file = open(dictionary_dir + file ,'r',encoding='utf-8')
    for line in file.readlines():
        line = line.split()
        words.extend(line)
    word_count.append(len(words))
print("Merge finished , merged {0} diectionaries , word count of each epoch = {1} ".format(str(len(word_count)),str(word_count)))


dict1_out = open(dictionary_dir + "dict_merged.txt", 'w', encoding='utf-8')

for word in words:
    dict1_out.write(word+" ")
dict1_out.close()
