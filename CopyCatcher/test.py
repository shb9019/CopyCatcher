from gensim.models import doc2vec

def myhash(obj):
    return hash(obj) % (2 ** 32)    

model = doc2vec.Doc2Vec(hashfxn=myhash)

#Load the model we trained earlier
model = doc2vec.Doc2Vec.load("../classifier/doc2vec/Doc2VecTaggedDocs")

# To get vector of a word if it exists in corpus
print(model["math"])

# doesnt_match function tries to deduce which word in a set is most
# dissimilar from the others
# print(model.doesnt_match("geometry neutron math science".split()) + '\n')
# print(model.doesnt_match("paris berlin london austria".split()) + '\n')

list_of_words = ["this", "is", "a", "new","unseen", "sentence"]
inferred_embedding = model.infer_vector(list_of_words)

print(inferred_embedding)

# most_similar(): returns the score of the most similar words based on the criteria
# Find the top-N most similar words. Positive words contribute positively towards the
# similarity, negative words negatively.
# print("most similar:")
# print(model.most_similar(positive=['mountain', 'king'], negative=['man'], topn=10))
# print()
# print(model.docvecs.most_similar("0"))
