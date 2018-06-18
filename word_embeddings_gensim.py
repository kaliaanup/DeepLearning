'''
Created on Apr 3, 2018

@author: kaliaanup
'''
#from gensim.models import word2vec
#from gensim.models.word2vec import Word2Vec
import gensim
import pickle

with open("data/word_embeddings/male_blog_list.txt","rb") as male_file:
    male_posts= pickle.load(male_file)

with open("data/word_embeddings/female_blog_list.txt","rb") as female_file:
    female_posts= pickle.load(female_file)
    
print(len(female_posts))
print(len(male_posts))

filtered_male_posts = list(filter(lambda p: len(p) > 0, male_posts))
filtered_female_posts = list(filter(lambda p: len(p) > 0, female_posts))
posts = filtered_female_posts + filtered_male_posts

print posts
print map(lambda x: x.split(), posts[:100])

# print(len(filtered_female_posts), len(filtered_male_posts), len(posts))
# 
#size corresponds to the NN layers, which correspond to the degree of freedom the training algorithm has: 
#min count is to prune uninteresting data default value is 5
w2v = gensim.models.Word2Vec(size=200, min_count=1)
w2v.build_vocab(map(lambda x: x.split(), posts[:100]), )

#print w2v.Vocab

print w2v.similarity('I', 'My')
print w2v.similarity('ring', 'husband')