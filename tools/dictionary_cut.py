import os
dictionary_dir = "../dictionaries/"
import random
def split_list(lst,cutTimes): #cutTimes 同一个列表随机切割几次
    result = []
    result.append(lst)
    for i in range(cutTimes):
        total = len(lst)
        prev_index = 0
        index = 0
        while(index < total):
            index = random.randint(prev_index,total)
            if prev_index == index:
                index += 1
            result.append(lst[prev_index:index])
            prev_index = index
    return result

for dictionary in os.listdir(dictionary_dir):
    if dictionary.endswith("_cut.txt"):
        raise Exception("Already cutted diectionary exists.Please remove it first.")
    if not dictionary.endswith("_merged.txt"):
        continue
    dict1 = open(dictionary_dir+dictionary,'r',encoding='utf-8')
    dict1_out = open(dictionary_dir+dictionary.split(".")[0]+"_cut.txt",'w',encoding='utf-8')
    line = dict1.readline().split()
    for word in line:
        result = split_list(word,cutTimes=3)
        for r in result:
            dict1_out.write(r+" ")
    dict1.close()
    dict1_out.close()