"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
# title_list = title_list [14930:25255]
train = pd.read_csv('resources/data/train.csv')
movies = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')

# Data Modifying
tr = train.copy()
tr = tr.drop('timestamp', axis=1)
merged = pd.merge(tr,movies,on='movieId') # Merging the dataframes
merge = merged.copy()
merge = merged.drop('genres', axis=1)

# new_df = pd.DataFrame()
# for i in title_list:
#     new_df = new_df.append(merge[merge['title'] == i])

merge = merge[:500000] # Slicing the data so that there is less computational power required
m = merge.pivot_table(index=['userId'],columns=['title'],values='rating') # piviting the table into a matrix

# Creating a new list of the titles, to be the ones that are referenced in the model itself
new_title_list = []
for i in m:
    new_title_list.append(i)

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    st.write('----------------------------------------------------------')
    st.title('Movie Recommendation Engine')
    st.write('----------------------------------------------------------')
    page_options = ["Recommender System","Solution Overview", "Insights"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        st.info("See **Solution Overview** and **Insights** pages for more information on modelling and analysis")
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',new_title_list[:1000])
        movie_2 = st.selectbox('Second Option',new_title_list[1:1000])
        movie_3 = st.selectbox('Third Option',new_title_list[2:1000])
        # movie_1 = st.selectbox('First Option',title_list[14930:15200])
        # movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        # movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [(movie_1,5),(movie_2,5),(movie_3,5)] # Added ratings, for more efficient use in the model

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                # try:
                with st.spinner('Crunching the numbers...'): # spinner just for something to happen during loading time
                    userRatings = m.dropna(thresh=10, axis=1).fillna(0,axis=1) # dropping and filling NaN values
                    corrMatrix = userRatings.corr(method='pearson') # creating a correlation Matrix
                    def get_similar(movie_name,rating=5): # Function for retriving similar movies based off correlation
                        similar_ratings = corrMatrix[movie_name]*(rating-2.5)
                        similar_ratings = similar_ratings.sort_values(ascending=False)
                        return similar_ratings
                    similar_movies = pd.DataFrame() # creating an empty Dataframe
                    for movie,rating in fav_movies: # Filling the empty dataframe, and extracting from it
                        similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)
                    recc_movies = similar_movies.sum().sort_values(ascending=False).head(14)[3:13] #summing and sorting DF, also slicing for no repeats
                    count = 1
                    st.markdown('## Top 10 Recommendations based on your movie picks:')
                    for key, value in dict(recc_movies).items(): # Displaying the output
                        st.info(str(count) + '. ' + str(key))
                        count += 1
                # except:
                #     st.error("Oops! Looks like this algorithm does't work.\
                #               We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown("## Content based filtering")
        st.write("Describe content based algo")
        st.markdown('## Collaborative based filtering')
        st.write('Describe collaborative algo here')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Insights":
        st.title('Our Insights and EDA')
        st.image('resources/imgs/image_insightpage.jpg', width=500)
        st.markdown("## Introduction")
        st.info("""Recommendation sysytems are an integral part to any online user based service platform. In short an alogrithim is 
                    created that will reccomened you items (eg: movies) based on your history with past items.
                    For this challenge we are tasked to create a movie recommendation system, There were two paths that we could take. 
                    Content Based Filtering and Collaborative Filtering, We chose to try out both in which we will get further into 
                    this notebook""")
        st.markdown('## Insteresting insights and EDA')
        st.write('**Top 5 Movies with the Highest Ratings:**')
        st.markdown("""
                    1. Final Recourse (2013)\n
                    2. Lady and the Tramp (2019)\n
                    3. Virasat (1997)\n
                    4. Committed (2014)\n
                    5. Sole Proprietor (2016)\n  """)
        st.write("\n**Distribution for Number of Ratings**")
        st.image('resources/imgs/rating_distibution.png', width=400)
        st.image('resources/imgs/distribution_graph.png', width=400)
        st.info("""From this we can see that people are more likely to give a movie an * Average * rating more than a great or awful rating.
        This makes sense because most movies are average and not every movie is great or awful. This could also lead us to belive that there may be certain biasy around our data.""")

if __name__ == '__main__':
    main()
