import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict


def get_top_n(predictions, n=5):
    """Return the top N recommendations for each user from a set of predictions."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


def get_top_recommendations_for_game(game_title, num_recommendations=5, chunk_size=10000):
    # First, find the app_id of the game title
    games_df = pd.read_csv('dataset/games.csv', usecols=['app_id', 'title'])
    try:
        game_app_id = games_df.loc[games_df['title'].str.lower() == game_title.lower(), 'app_id'].iloc[0]
    except IndexError:
        return f"No games found with title '{game_title}'."
    
    # Reading in chunks: filter out reviews for the specified game
    reader = Reader(rating_scale=(0, 1))
    recommendations_chunks = pd.read_csv('dataset/recommendations.csv', chunksize=chunk_size)
    relevant_data = []
    for chunk in recommendations_chunks:
        chunk = chunk[chunk['app_id'] == game_app_id]
        relevant_data.append(chunk)
    relevant_reviews_df = pd.concat(relevant_data, ignore_index=True)
    if relevant_reviews_df.empty:
        return f"No recommendations found for game '{game_title}'."
    
    data = Dataset.load_from_df(relevant_reviews_df[['user_id', 'app_id', 'is_recommended']], reader)
    
    # Build a collaborative filtering model to predict ratings for games other than the specified game
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = get_top_n(predictions, n=num_recommendations)
    
    # Pairing game IDs with game titles and user predictions
    top_game_titles = defaultdict(list)
    for uid, user_ratings in top_n.items():
        top_game_titles[uid] = [(games_df[games_df['app_id'] == iid]['title'].iloc[0], est) for (iid, est) in user_ratings]

    # Select the most frequently recommended games that are not the specified game
    recommended_games = defaultdict(int)
    for user_ratings in top_game_titles.values():
        for title, _ in user_ratings:
            recommended_games[title] += 1
    sorted_recommendations = sorted(recommended_games.items(), key=lambda x: x[1], reverse=True)
    top_recommendation_titles = [title for title, _ in sorted_recommendations if title.lower() != game_title.lower()]
    
    return top_recommendation_titles[:num_recommendations]


top_recommendations = get_top_recommendations_for_game('The Elder Scrolls V: Skyrim Special Edition')
print(f"Your recommendations: {top_recommendations}")
