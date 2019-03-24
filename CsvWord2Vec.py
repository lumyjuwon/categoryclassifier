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
    category = i[4]  # CSV에서 category Column으로 변경
    category_number_dic = {'IT과학': 0, '경제': 1, '정치': 2, 'e스포츠': 3, '골프': 4, '농구': 5, '배구': 6, '야구': 7, '일반 스포츠': 8, '축구': 9, '사회': 10, '생활문화': 11}
    for key in category_number_dic.keys():
        if category == key:
            all_temp.append(category_number_dic.get(category))
            break
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
