from konlpy.tag import Twitter
from gensim.models import Word2Vec
import csv

"""
@author: lumyjuwon
"""

twitter = Twitter()

file = open("Article_shuffled.csv", 'r', encoding='euc-kr')
line = csv.reader(file)
token = []
embeddingmodel = []

for i in line:
    sentence = twitter.pos(i[0], norm=True, stem=True)
    temp = []
    temp_embedding = []
    all_temp = []
    for k in range(len(sentence)):
        temp_embedding.append(sentence[k][0])
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    all_temp.append(temp)
    embeddingmodel.append(temp_embedding)
    if i[3] == "IT과학":
        all_temp.append(0)
    elif i[3] == "경제":
        all_temp.append(1)
    elif i[3] == "정치":
        all_temp.append(2)
    elif i[3] == "e스포츠":
        all_temp.append(3)
    elif i[3] == "골프":
        all_temp.append(4)
    elif i[3] == "농구":
        all_temp.append(5)
    elif i[3] == "배구":
        all_temp.append(6)
    elif i[3] == "야구":
        all_temp.append(7)
    elif i[3] == "일반 스포츠":
        all_temp.append(8)
    elif i[3] == "축구":
        all_temp.append(9)
    elif i[3] == "사회":
        all_temp.append(10)
    elif i[3] == "생활문화":
        all_temp.append(11)
    token.append(all_temp)
print("토큰 처리 완료")


embeddingmodel = []
for i in range(len(token)):
    temp_embeddingmodel = []
    for k in range(len(token[i][0])):
        temp_embeddingmodel.append(token[i][0][k])
    embeddingmodel.append(temp_embeddingmodel)
embedding = Word2Vec(embeddingmodel, size=300, window=5, min_count=10, iter=5, sg=1, max_vocab_size=360000000)
embedding.save('post.embedding')
