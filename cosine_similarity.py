from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]

# count number of features(words) for every entry in the text array
cv = CountVectorizer()


count_matrix = cv.fit_transform(text)


print(count_matrix.toarray())

# get cosine similarity
similarity_scores = cosine_similarity(count_matrix)

print(similarity_scores)


