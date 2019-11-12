import random
MAX = 10000
MIN = 0
dictionary_dir = "../dictionaries/"
COUNT = 1200000
ratio_digits = 1 #总输出=COUNT*(1+ratio_digits)
dict1_out = open(dictionary_dir + "numbers.txt", 'w', encoding='utf-8')

def gen_digits():
    ret = ""
    k = random.random()
    if ratio_digits - random.random() > 0 and k < 0.2 :
        if random.random() >= 0.8:
            ret+= ("%.2f" % (random.random()*100)+"%")
        else:
            ret+= ("%.2f" % (random.random()*100))
    elif ratio_digits - random.random() > 0 and k >= 0.6:
        if random.random() >= 0.8:
            ret+= ("%.1f" % (random.random()*100)+"%")
        else:
            ret+= ("%.1f" % (random.random()*100))
    else:
        p = random.random()
        if p >= 0.3 :
            ret += str(int(random.random()*10))
        elif p >= 0.2:
            ret += str(int(random.random()*10)) + '%'
        else:
            ret += str(int(random.random()*100)) + '%'




    return ret


for i in range(COUNT):
    dict1_out.write(str(random.randint(MIN,MAX))+" ")
    dict1_out.write(gen_digits()+" ")
dict1_out.close()