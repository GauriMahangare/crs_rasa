# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import json
from typing import Any, Text, Dict, List

import requests

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import random
# Models for content filtering based on cosine similarity
similarity = pickle.load(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/content_filtering/cosine_similarity.pkl", 'rb'))
movie_tag_df = pickle.load(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/content_filtering/movies.pkl", 'rb'))
# Models for content filtering for top trending movies
top_trending_movies_df = pd.read_csv(open(
    "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/data/processed/outputs/top_trending_content.csv", 'r'))
genres_df = pd.read_csv(open(
    "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/data/processed/entity/entity_movie_genres.csv", 'r'))
# Models for collaborative filtering item-item similarity
collab_similarity = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/collab_filtering_item_item_similarity/collab_similarity.pkl", 'rb'))
collab_ratings_pt_indexes = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/collab_filtering_item_item_similarity/collab_ratings_pt_indexes.pkl", 'rb'))
collab_ratings_pt = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/collab_filtering_item_item_similarity/collab_ratings_pt.pkl", 'rb'))
movie_list_full_df = pd.read_csv(open(
    "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/data/processed/movieLense/movies_combined_cleaned_title.csv", 'r'))

# Models for collaborative filtering user-user similarity
collab_u2u_similarity = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/collab_filtering_user_user_similarity/collab_similarity.pkl", 'rb'))
collab_u2u_ratings_pt = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/collab_filtering_user_user_similarity/collab_ratings_pt.pkl", 'rb'))
users_list_df = pd.read_csv(open(
    "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/data/processed/movieLense/users.csv", 'r'))
ratings_df = pd.read_csv(open(
    "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/data/processed/movieLense/ratings_short.csv", 'r'))

# Models for DNN
# DNN_ratings_model = tf.saved_model.load(
#     "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/DS/crs_ds/models/DNN_ratings_prediction/")
DNN_ratings_model = tf.keras.models.load_model(
    "/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/DNN_ratings_prediction/cf_dnn_model")

DNN_ratings_model_df = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/DNN_ratings_prediction/dnn_ratings_pred_df.pkl", 'rb'))
DNN_movie2movie_encoded = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/DNN_ratings_prediction/dnn_movie2movie_encoded.pkl", 'rb'))
DNN_user2user_encoded = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/DNN_ratings_prediction/dnn_user2user_encoded.pkl", 'rb'))
DNN_movie_encoded2movie = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/DNN_ratings_prediction/dnn_movie_encoded2movie.pkl", 'rb'))

# models for matrix factorisation
U_reg = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/matrix_factorisation/user_embedding.pkl", 'rb'))
V_reg = pd.read_pickle(
    open("/Users/gauridhumal/Development Projects/UOL-PROJECTs/CRS/crs_ds/models/matrix_factorisation/item_embedding.pkl", 'rb'))

imdb_url = "https://www.imdb.com/title/"


def get_user_profile_by_id(user_id):
    # Call your API here
    url = "http://localhost:8000/api/users-profile-by-user/?format=json&user_id=" + user_id
    payload = {}
    headers = {
        'Authorization': 'Token e269631b14eb69bdd2ff3c8522f0b581640e198a'
    }
    try:
        response = requests.request(
            "GET", url, headers=headers, data=payload, timeout=3)
        print(response)
        return response
    except Exception as e:
        return response({'Error calling API - get_user_profile_by_id': str(e)})


def update_user_profile_api(user_data):
    url = "http://localhost:8000/api/users-profile/"
    payload = json.dumps(user_data)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Token e269631b14eb69bdd2ff3c8522f0b581640e198a'
    }
    try:
        response = requests.request(
            "POST", url, headers=headers, data=payload, timeout=3)
        return response
    except Exception as e:
        return response({'Error calling API-1 - update_user_profile': str(e)})


def update_user_profile(user_id, genre_liked):

    response = get_user_profile_by_id(user_id)
    if response.status_code == 200:
        data = response.json()
        # Extract relevant information from the response and send it back to the user
        user_data = data['results'][0]
        user_data['movie_pref_1'] = genre_liked
        # Call your API  to update
        return update_user_profile_api(user_data)
    else:
        return response({'Error calling API-2 - update_user_profile': str(e)})


def remove_special_characters(text):
    # Define a pattern to keep only alphanumeric characters
    pattern = re.compile(r'[^a-zA-Z0-9\s]')

    # Use the pattern to replace non-alphanumeric characters with an empty string
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

# Search for the value in the list of tuples


def find_index(movie):
    search_value = movie_list_full_df[movie_list_full_df['cleaned_title'].isin(
        [movie])]['imdb_id'].values[0]
    found_tuples = [
        tup for tup in collab_ratings_pt_indexes if search_value in tup]

    # Display the result
    if found_tuples:
        print(f"Found tuples containing '{search_value}':")
        for found_tuple in found_tuples:
            print(found_tuple)
            print(found_tuple[0])
        return found_tuples[0]
    else:
        print(f"No tuples containing '{search_value}' found.")
        return None


def compute_scores(query_embedding, item_embeddings, measure):
    """Computes the scores of the candidates given a query.
    Args:
      query_embedding: a vector of shape [k], representing the query embedding.
      item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
      measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
      scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    q = query_embedding
    I = item_embeddings
    if measure == "COSINE":
        I = I / np.linalg.norm(I, axis=1, keepdims=True)
        q = q / np.linalg.norm(q)
    scores = q.dot(I.T)
    return scores


def user_recommendations(measure, query_embedding, item_embeddings):
    scores = compute_scores(query_embedding, item_embeddings, measure)
    score_key = measure+"_" + 'score'
    df = pd.DataFrame({
        'score_key': list(scores),
        'movie_id': movie_list_full_df['imdb_id'],
        'titles': movie_list_full_df['title'],
        'genres': movie_list_full_df['genre_tags'],
    })
    return df


def movie_neighbours(title_substring, measure, k, query_embedding, item_embeddings):
    # Select the most matching title
    print(movie_list_full_df.columns)
    title_substring = remove_special_characters(
        title_substring).lower().replace(" ", "")
    print(title_substring)
    ids = movie_list_full_df[movie_list_full_df['cleaned_title'].str.contains(
        title_substring)].index.values
    print(ids)
    titles = movie_list_full_df.iloc[ids]['title'].values
    if len(titles) == 0:
        # raise ValueError("Found no movies with title %s" % title_substring)
        other_matching_titles = "Found no movies with title %s" % title_substring
        df = pd.DataFrame()
        return df, other_matching_titles
    else:
        print("Nearest neighbors of : %s." % titles[0])
        print("[Found more than one matching movie. Other candidates: {}]".format(
            ", ".join(titles[1:])))
        other_matching_titles = ", ".join(titles[0:])
        movie_id = ids[0]

        query_embedding = query_embedding[movie_id]
    # Calculating dot matrix this the most matched movie with other movie embeddings to find the other matching movies
        scores = compute_scores(query_embedding, item_embeddings, measure)

        score_key = measure + "_" + "score"
        # df['score_key'] = list(scores),
        # df['movie_id'] = movie_list_full_df['imdb_id'],
        # df['titles'] = movie_list_full_df['title'],
        # df['genres'] = movie_list_full_df['genre_tags'],
        df = pd.DataFrame({
            score_key: list(scores),
            'movie_id': movie_list_full_df['imdb_id'],
            'titles': movie_list_full_df['title'],
            'genres': movie_list_full_df['genre_tags'],
        })
        print(df)
        print(type(df))
        print(other_matching_titles)
        return df, other_matching_titles


def return_searched_results(searched_movies):
    search_list = []
    nbr_of_movies = 0
    for index, row in searched_movies.iterrows():
        nbr_of_movies = nbr_of_movies + 1
        print(
            f"Index: {index}, Values: {row['ml_id']}, {row['title']}, {row['imdb_id']}")
        new_list = {'index': index,
                    'imdb_id': row['imdb_id'],
                    'title': row['title'],
                    'tags': row['tags']}
        search_list.append(new_list)
        if nbr_of_movies > 6:
            break
    return search_list


class ActionRecommendPupularContent(Action):
    """Returns popular content

    Args:
        Action (_type_): takes genre input

    Returns:
        _type_: Returns top trending movies for the given genre
    """

    def name(self) -> Text:
        return "recommend_popular_content"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            u_genre = tracker.latest_message['entities'][0]['value']
        except:
            dispatcher.utter_message(
                text="[ERROR]I'm sorry, I didn't quite understand your message. Could you please rephrase or provide more details so I can assist you better? Thank you!")
            return []
        else:
            top_trending_df = top_trending_movies_df[(
                top_trending_movies_df['titleType'] == 'movie') & (top_trending_movies_df['genres'].str.contains(u_genre))].head(10)
            top_trending_df = top_trending_df.sort_values(
                'weighted_rating', ascending=False)
            if top_trending_df.empty:
                dispatcher.utter_message(
                    text="[ERROR]No trending movies found for {genre}".format(genre=u_genre))
                return []
            else:
                top_trending = []
                for index, row in top_trending_df.iterrows():
                    new_list = {'imdb_id': row['tconst'],
                                'title': row['primaryTitle'],
                                'weighted_rating': row['weighted_rating'], }
                    top_trending.append(new_list)
                dispatcher.utter_message(
                    text="Here are top trending movies in {genre} category".format(genre=u_genre))
                concatenated_titles_with_ids = ""
                concatenated_titles_with_ids = ''.join(
                    f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in top_trending)
                dispatcher.utter_message(text=concatenated_titles_with_ids)
                return []


class ActionSearchMovies(Action):
    """

    Args:
        Action (_type_):

    Returns:
        _type_:
    """

    def name(self) -> Text:
        return "search_movies"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            search_string_orig = tracker.latest_message['entities'][0]['value']
        except:
            dispatcher.utter_message(
                text="[ERROR-1]I'm sorry, I didn't quite understand your message. Could you please rephrase or provide more details so I can assist you better? Thank you!")
            return []
        else:
            print(search_string_orig)
            search_string = remove_special_characters(
                search_string_orig.lower().strip()).replace(" ", "")
            searched_movies = movie_tag_df[movie_tag_df['tags'].str.contains(
                search_string)]
            if searched_movies.empty:
                print("String not found in tag - " + search_string)
                searched_movies = movie_tag_df[movie_tag_df['cleaned_title'].str.contains(
                    search_string)]
                if searched_movies.empty:
                    search_list = []
                    print("String not found in title - " + search_string)
                    dispatcher.utter_message(
                        text="[ERROR-2]I'm sorry, I didn't quite understand your message. Could you please rephrase or provide more details so I can assist you better? Thank you!")
                    return []
                else:
                    search_list = return_searched_results(searched_movies)
                    dispatcher.utter_message(
                        text="Here's what I have found for '{text}'".format(text=search_string_orig))
                    concatenated_titles_with_ids = ""
                    concatenated_titles_with_ids = ''.join(
                        f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in search_list)
                    dispatcher.utter_message(text=concatenated_titles_with_ids)
                    return []
            else:
                search_list = return_searched_results(searched_movies)
                dispatcher.utter_message(
                    text="Here's what I have found for '{text}'".format(text=search_string_orig))
                concatenated_titles_with_ids = ""
                concatenated_titles_with_ids = ''.join(
                    f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in search_list)
                dispatcher.utter_message(text=concatenated_titles_with_ids)
                return []


class ActionRecommendMovieItem2ItemSearch(Action):
    """
    Implemented using matrix factorisation nearest neighbours
    Args:
        Action (_type_): Takes movies title and finds its
        nearest neighbours by doing cosine similarity score

    Returns:
        _type_: List of movies similar to provides
    """

    def name(self) -> Text:
        return "recommend_movie_item2item_search"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            title_substring = tracker.latest_message['entities'][0]['value']
        except:
            dispatcher.utter_message(
                text="[ERROR-1]I'm sorry, I didn't quite understand your message. Could you please rephrase or provide more details so I can assist you better? Thank you!")
            return []
        else:
            measure = "COSINE"
            score_key = measure+"_" + 'score'
            k = 6  # Number of movies to be recommended
            query_embedding = V_reg.numpy()
            item_embeddings = V_reg.numpy()
            df, other_matching_titles = movie_neighbours(
                title_substring, measure, k, query_embedding, item_embeddings)
            m_list = []
            if df.empty:
                print("Title not found in the database - " + title_substring)
                dispatcher.utter_message(
                    text="[ERROR-2]Title not found in the movie DB. Try again with new search")
                return []
            else:
                recom_movie_df = df.sort_values(
                    [score_key], ascending=False).head(k)
                for index, row in recom_movie_df.iterrows():
                    new_list = {'imdb_id': row['movie_id'],
                                'title': row['titles'],
                                'genres': row['genres'],
                                }
                    m_list.append(new_list)
                dispatcher.utter_message(
                    text="Here's what I have found for '{text}'".format(text=title_substring))
                concatenated_titles_with_ids = ""
                concatenated_titles_with_ids = ''.join(
                    f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in m_list)
                dispatcher.utter_message(text=concatenated_titles_with_ids)
                return []


class ActionSurpriseRecommendations(Action):
    """

    Args:
        Action (_type_): We will use matrix factorisation to surprise the user.
        We will exclude all the movies which this user has already rated.

    Returns:
        _type_: list of movies
    """

    def name(self) -> Text:
        return "surprise_recommendations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        measure = "COSINE"
        # user_id = random.randint(1, 513)
        user_id = int(tracker.sender_id)
        # user_id = int(user_id_input)
        exclude_rated = "Yes"
        k = 6  # Number of movies to be recommended
        query_embedding = V_reg.numpy()[user_id]
        item_embeddings = V_reg.numpy()
        df = user_recommendations(
            measure, query_embedding, item_embeddings)
        rated_movies = ratings_df[ratings_df.user_id ==
                                  user_id]["movie_id"].values
        if exclude_rated == "Yes":
            # remove movies that are already rated
            df = df[df.movie_id.apply(
                lambda movie_id: movie_id not in rated_movies)]
        recom_movie_df = df.sort_values(
            ["score_key"], ascending=False).head(k)
        m_list = []
        for index, row in recom_movie_df.iterrows():
            new_list = {'imdb_id': row['movie_id'],
                        'title': row['titles'],
                        'genres': row['genres']}
            m_list.append(new_list)
        concatenated_titles_with_ids = ""
        concatenated_titles_with_ids = ''.join(
            f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in m_list)
        dispatcher.utter_message(
            text="Here are your surprised recommendations...")
        dispatcher.utter_message(text=concatenated_titles_with_ids)
        return []


class ActionRecommendCollabFilteringU2U(Action):
    """

    Args:
        Action (_type_):

    Returns:
        _type_:
    """

    def name(self) -> Text:
        return "recommend_collaborative_filtering_u2u"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        measure = "COSINE"
        # user_id = random.randint(1, 513)
        user_id = int(tracker.sender_id)
        # user_id = int(user_id_input)
        exclude_rated = "Yes"
        k = 6  # Number of movies to be recommended
        query_embedding = V_reg.numpy()[user_id]
        item_embeddings = V_reg.numpy()
        df = user_recommendations(
            measure, query_embedding, item_embeddings)
        rated_movies = ratings_df[ratings_df.user_id ==
                                  user_id]["movie_id"].values
        if exclude_rated == "Yes":
            # remove movies that are already rated
            df = df[df.movie_id.apply(
                lambda movie_id: movie_id not in rated_movies)]
        recom_movie_df = df.sort_values(
            ["score_key"], ascending=False).head(k)
        m_list = []
        for index, row in recom_movie_df.iterrows():
            new_list = {'imdb_id': row['movie_id'],
                        'title': row['titles'],
                        'genres': row['genres']}
            m_list.append(new_list)
        concatenated_titles_with_ids = ""
        concatenated_titles_with_ids = ''.join(
            f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in m_list)
        dispatcher.utter_message(
            text="Here's what others are watching..")
        dispatcher.utter_message(text=concatenated_titles_with_ids)
        return []


class ActionRecommendPersonalisedRecommendationDNN(Action):
    """

    Args:
        Action (_type_):

    Returns:
        _type_:
    """

    def name(self) -> Text:
        return "recommend_personalised_recommendation_DNN"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # user_id = random.randint(1, 513)
        user_id = int(tracker.sender_id)
        movies_watched_by_user = DNN_ratings_model_df[DNN_ratings_model_df.user_id == user_id]
        movies_not_watched = movie_list_full_df[
            ~movie_list_full_df["ml_id"].isin(movies_watched_by_user.movie_id.values)]["ml_id"]
        # extract movies from not watch list that other users have rated
        movies_not_watched = list(set(movies_not_watched).intersection(
            set(DNN_movie2movie_encoded.keys())))
        # extract the index of these
        movies_not_watched = [
            [DNN_movie2movie_encoded.get(x)] for x in movies_not_watched]
        # get user index
        user_encoder = DNN_user2user_encoded.get(user_id)
        user_input = np.array([[user_encoder]] * len(movies_not_watched))
        movies_input = np.array(movies_not_watched)
        # predict user ratings on unseen movies
        user_ratings = DNN_ratings_model.predict(
            [user_input, movies_input]).flatten()
        print(user_ratings)
        # Select top 10 movies with highest ratings
        top_ratings_indices = user_ratings.argsort()[-10:][::-1]
        # Get the original movie ids for recommended movies
        recommended_movie_ids = [DNN_movie_encoded2movie.get(
            movies_not_watched[x][0]) for x in top_ratings_indices]
        # Movies recommended to this user
        recomm_movies = movie_list_full_df[movie_list_full_df["ml_id"].isin(
            recommended_movie_ids)]
        m_list = []
        for index, row in recomm_movies.iterrows():
            new_list = {'imdb_id': row['imdb_id'],
                        'title': row['title']}
            m_list.append(new_list)
        print(m_list)
        concatenated_titles_with_ids = ""
        concatenated_titles_with_ids = ''.join(
            f"<li><a href={imdb_url}{movie['imdb_id']} target='_blank'> {movie['title']} </a></li>" for movie in m_list)
        dispatcher.utter_message(
            text="Here's what I have found based on your liking")
        dispatcher.utter_message(text=concatenated_titles_with_ids)
        return []


class ActionRetrieveCurrentPreferences(Action):
    """

    Args:
        Action (_type_): Retrieves current user preferecens on the movies

    Returns:
        _type_: List of genres
    """

    def name(self) -> Text:
        return "retrieve_current_preferences"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # user_id = random.randint(1, 513)
        user_id = tracker.sender_id
        response = get_user_profile_by_id(user_id)
        if response.status_code == 200:
            data = response.json()
            # Extract relevant information from the response and send it back to the user
            first_instance = data['results'][0]
            concatenated_preferences = ""
            concatenated_preferences = ''.join(
                f"<li>{first_instance['movie_pref_1']}</li><li>{first_instance['movie_pref_2']}</li><li>{first_instance['movie_pref_3']}</li>")
            dispatcher.utter_message(
                text="Here are your current preferences against your profile")
            dispatcher.utter_message(text=concatenated_preferences)
        else:
            dispatcher.utter_message(text="Failed to fetch data from API.")
        return []


class ActionUpdateCurrentPreferences(Action):
    """

    Args:
        Action (_type_): Retrieves current user preferecens on the movies

    Returns:
        _type_: List of genres
    """

    def name(self) -> Text:
        return "update_all_preferences"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # user_id = random.randint(1, 513)
        user_id = tracker.sender_id
        print(tracker.latest_message['entities'][0]['value'])
        genre_list = tracker.latest_message['entities'][0]['value']
        # genre_list = genre_list.replace(" ", "")
        genre1, genre2, genre3 = (genre.strip()
                                  for genre in genre_list.split(","))

        response = get_user_profile_by_id(user_id)

        if response.status_code == 200:
            data = response.json()
            user_data = data['results'][0]
            user_data['movie_pref_1'] = genre1
            user_data['movie_pref_2'] = genre2
            user_data['movie_pref_3'] = genre3
            response = update_user_profile_api(user_data)
            if response.status_code == 200:
                print(response.text)
                dispatcher.utter_message(
                    text="Thanks, I have updated your preferences.")
            else:
                dispatcher.utter_message(
                    text="Failed to fetch data from API2.")
        else:
            dispatcher.utter_message(text="Failed to fetch data from API1.")
        return []


class ActionUpdateSingleGenreLike(Action):
    """

    Args:
        Action (_type_): Updates preference which user likes

    Returns:
        _type_: List of genres
    """

    def name(self) -> Text:
        return "update_single_genre_like"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_id = tracker.sender_id
        genre_liked = tracker.latest_message['entities'][0]['value']
        response = update_user_profile(user_id, genre_liked)
        if response.status_code == 200:
            print(response.text)
            dispatcher.utter_message(
                text="Thanks, I will remember that.")
        else:
            dispatcher.utter_message(text="Failed to fetch data from API.")
        return []


class ActionUpdateSingleGenreDisLike(Action):
    """

    Args:
        Action (_type_): Removes prefereneces which user does not like

    Returns:
        _type_: List of genres
    """

    def name(self) -> Text:
        return "update_single_genre_dislike"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_id = tracker.sender_id
        genre_disliked = tracker.latest_message['entities'][0]['value']
        genre_disliked_formatted = genre_disliked.replace(" ", "").lower()

        response = get_user_profile_by_id(user_id)
        if response.status_code == 200:
            data = response.json()
            user_data = data['results'][0]
            if user_data['movie_pref_1'].lower() == genre_disliked_formatted:
                user_data['movie_pref_1'] = ""
            if user_data['movie_pref_2'].lower() == genre_disliked_formatted:
                user_data['movie_pref_2'] = ""
            if user_data['movie_pref_3'].lower() == genre_disliked_formatted:
                user_data['movie_pref_3'] = ""
            response = update_user_profile_api(user_data)
            if response.status_code == 200:
                print(response.text)
                dispatcher.utter_message(
                    text="Thanks, I will remember that.")
            else:
                dispatcher.utter_message(
                    text="Failed to fetch data from API2.")
        else:
            dispatcher.utter_message(text="Failed to fetch data from API1.")
        return []
