import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


df = pd.read_csv("movie_dataset.csv")
print(df.columns)

features = ['keywords','cast','genres','director']

#fill all na with empty string
for feature in features:
	df[feature] = df[feature].fillna('')

def combine(row):
	try:
		return row['keywords'] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
	except:
		print("error:",row)
#passes each row vertically
df["combine_features"] = df.apply(combine,axis=1)
print("Combined features:",df["combine_features"].head())

#Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combine_features"])

#Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

#Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movie = list(enumerate(cosine_sim[movie_index]))

# Get a list of similar movies in descending order of similarity score
sorted_movies = sorted(similar_movie,key=lambda x:x[1],reverse=True)

#Print titles of first 50 movies
i = 0
for movie in sorted_movies:
	print(get_title_from_index(movie[0]))
	i+=1
	if i>50:
		break