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

##Use images

from PIL import Image
Home = Image.open('resources/imgs/Home.png')

# images on how it works page
How = Image.open('resources/imgs/How.png')

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
train = pd.read_csv('resources/data/train.csv')
movies = pd.read_csv('resources/data/movies.csv')

# Data Modifying
tr = train.copy()
tr = tr.drop('timestamp', axis=1)
merged = pd.merge(tr,movies,on='movieId') # Merging the dataframes
merge = merged.copy()
merge = merged.drop('genres', axis=1)

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
    page_options = ["Home","How it works","Recommender System","Solution Overview","EDA Page"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    #
    #
    # Building the home page    
    if page_selection == "Home":
        st.title("Movie Recommender")
        st.subheader("Welcome!")
        st.markdown("This web app recommends movies based on similar or related to movies a user selects.")        
        st.image(Home, caption='', width=500)        
        st.subheader("Why recommender systems")
        st.markdown("Any streaming platforms is built around lessening one’s time trying to decide which movie to watch. We supply users with relative content to watch taking into consideration their values and ideals. We would like to determine how people perceive streaming services and whether or not there is an issue that would be better rectified. This would help companies add to their market research efforts in gauging how their service recommendations may be improved.")
       
        
    # Building out the 'How it works' page
    
    if page_selection == "How it works":
        st.title("How it works")
##    	st.image(How, caption='', width=500)
##    	st.subheader("Navigation Bar")
        st.markdown("Home Screen  - Landing page, gives brief discription of the App")
    ##	st.image(Home_Page, width=500)
        st.markdown("How it works - Explains how the app works, from navigating on pages, to how efficient the model used is.")
    ##	st.image(How_to_Page, width=500)
        st.markdown("EDA          - Exploratory data analysis shows how we analysing the tweets data sets to summarize their main characteristics, using visuals. EDA is basically for displaying what the data can tell us beyond the formal modeling.")
    ##	    st.image(EDA_Page, width=500)
    ##	    st.image(Predictions_Page, width=500)
        st.subheader("App Usage")
        st.markdown("")
        st.subheader("Model Performance Evaluation")
        st.markdown("Model evaluation aims to estimate the generalization accuracy of a model on future (unseen) data. Methods for evaluating a model's performance use a test set (i.e data not seen by the model) to evaluate model performance. This evaluation shows total efficiency as scores.")
            
    # Building out the 'EDA' page   
    if page_selection == "EDA":
    	st.title("Exploratory Data Analysis")   
    	st.markdown("Analysis of the training data is an important step in understanding the data. A variety of analysis has been done on the training data. Select an option for more information.")


    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',new_title_list[:1000])
        movie_2 = st.selectbox('Second Option',new_title_list[:1000])
        movie_3 = st.selectbox('Third Option',new_title_list[:1000])
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
                try:
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
                        st.markdown('## Top 10 Recommendations based on your movie choices:')
                        for key, value in dict(recc_movies).items(): # Displaying the output
                            st.info(str(count) + '. ' + str(key))
                            count += 1
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.title("Content-based Filtering")
        st.write("Content-based filtering, also referred to as cognitive filtering, recommends items based on a comparison between the content of the compared items, in this case its movie content, and a user profile. The content of each movie is represented as a set of descriptors or terms. Generally, Content-based filtering, makes recommendations based on user preferences for product features")

        st.title("Collaborative-based Filtering")
        st.write("Collaborative filtering filters information by using the interactions and data collected by the system from other users. It's based on the idea that people who agreed in their evaluation of certain items are likely to agree again in the future. Collaborative filtering basically mimics user-to-user recommendations.")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
