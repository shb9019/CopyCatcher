from gensim.models import doc2vec

def myhash(obj):
    return hash(obj) % (2 ** 32)    

model = doc2vec.Doc2Vec(hashfxn=myhash)

#Load the model we trained earlier
model = doc2vec.Doc2Vec.load("../classifier/doc2vec/Doc2VecTaggedDocs")

# To get vector of a word if it exists in corpus
print(model["math"])

list_of_words = ["this", "is", "a", "new","unseen", "sentence"]
inferred_embedding = model.infer_vector(list_of_words)

print(inferred_embedding)
