import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def content_based_filtering(game_title, games_df, metadata_df, sample_size):
    # Take a sample of the whole dataset to avoid memory errors
    combined_features = games_df.merge(metadata_df, left_on='app_id', right_on='app_id', how='left')
    combined_features_sample = combined_features.sample(n=sample_size, replace=True)
    
    # Combine the game's title, description, and tags into a single text feature
    combined_features_sample['text'] = (combined_features_sample['title'] + " " +
                                        combined_features_sample['description'] + " " +
                                        combined_features_sample['tags'].apply(lambda tags: " ".join(tags)))
    
    # Create the TF-IDF Vectorizer and transform the combined text data
    tfidf_vectorizer = TfidfVectorizer(dtype=np.float32)
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_features_sample['text'])
    
    # Use TruncatedSVD to reduce the dimensionality of the TF-IDF matrix
    svd = TruncatedSVD()
    tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)
    
    # Calculate cosine similarity using the reduced matrix
    cosine_sim = cosine_similarity(tfidf_matrix_reduced, tfidf_matrix_reduced)
    
    # Get the index of the game that matches the title from the sampled dataset
    try:
        game_idx = combined_features_sample.index[combined_features_sample['title'] == game_title].tolist()[0]
    except IndexError:
        return []  # No game found
    
    # Get the pairwise similarity scores of all games with that game
    sim_scores = list(enumerate(cosine_sim[game_idx - combined_features_sample.index[0]]))

    # Sort the games based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar games
    sim_scores = sim_scores[1:6] # Skip the first one as it is the input game itself

    # Get the game indices
    game_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar games
    return combined_features_sample.iloc[game_indices]['title'].tolist()


SAMPLE_SIZE = 15_000

# Load the games.csv but sample it to a smaller size before merging if needed
games_df = pd.read_csv('dataset/games.csv').sample(n=SAMPLE_SIZE, replace=True)

# Load the games_metadata.json as done previously
# NOTE: Each line in the .json is an object, for some reason
metadata_list = []
with open('dataset/games_metadata.json', 'r') as f:
    for line in f:
        metadata_list.append(json.loads(line))
metadata_df = pd.DataFrame(metadata_list)

# Preprocessing.
games_duplicates = 0
games_duplicates = games_df.duplicated(subset="title").sum()

print(f"DUPLICATE GAMES: {games_duplicates}")

# Removing duplicates.
games_df = games_df.drop_duplicates(subset="title")
print(f"DUPLICATE GAMES NOW: {games_df.duplicated(subset='title').sum()}") 

# Calling and printing the recommendations.
recommendations = content_based_filtering('Batman', games_df, metadata_df, sample_size=SAMPLE_SIZE)
print(f"Your recommendations: {recommendations}")