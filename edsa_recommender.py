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
# train = pd.read_csv('resources/data/train.csv')
movies = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')

# Data Modifying
# tr = train.copy()
# tr = tr.drop('timestamp', axis=1)
ratings_df = ratings_df.drop('timestamp', axis = 1)

merged = pd.merge(ratings_df,movies,on='movieId') # Merging the dataframes
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
    st.write('----------------------------------------------------------')
    st.title('Movie Recommendation Engine')
    st.write('----------------------------------------------------------')
    page_options = ["Home", "Recommender System", "Info", "Solution Overview", "Insights"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Home":
        st.title("Home")
        st.subheader("Welcome!")
        st.markdown("This web app recommends movies based on similar or related to movies a user selects.")        
        st.image('resources/imgs/netflix_img.jpg',width=500)
        st.info('See ** Recommender Systems ** page to run the engine')        
        st.subheader("Why recommender systems")
        st.markdown("Any streaming platforms is built around lessening one’s time trying to decide which movie to watch. We supply users with relative content to watch taking into consideration their values and ideals. We would like to determine how people perceive streaming services and whether or not there is an issue that would be better rectified. This would help companies add to their market research efforts in gauging how their service recommendations may be improved.")


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

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            # Content Based options
            movie_1 = st.selectbox('First Option',title_list[14930:15200])
            movie_2 = st.selectbox('Second Option',title_list[25055:25255])
            movie_3 = st.selectbox('Third Option',title_list[21100:21200])
            fav_movies = [(movie_1),(movie_2),(movie_3)]

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
            # Collaborative Based Options
            movie_1_colab = st.selectbox('First Option',title_list[:1000])
            movie_2_colab = st.selectbox('Second Option',title_list[1:1000])
            movie_3_colab = st.selectbox('Third Option',title_list[2:1000])
            fav_movies_colab = [(movie_1_colab,5),(movie_2_colab,5),(movie_3_colab,5)] # Added ratings, for more efficient use in the model

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
                    for movie,rating in fav_movies_colab: # Filling the empty dataframe, and extracting from it
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
    
    if page_selection == "Info":
        st.title("Info")
        st.markdown("* ** Home Screen **  - Landing page, gives brief discription of the App")
        st.markdown("* ** Info ** - Explains how the app works, from navigating on pages, to how efficient the model used is.")
        st.markdown("* ** Recommender Systems **  - Main page that includes the movie recommender engine")
        st.markdown("* ** Insights **          - Exploratory data analysis shows how we analysing the data sets to summarize their main characteristics, using visuals. EDA is for displaying what the data can tell us beyond the formal modeling.")
        st.markdown("* ** Solution Overview ** - An overview on how the two different algorithims work")
        st.subheader("App Usage")
        st.markdown("Select the type of algorithm you want to use then select three farvorite movies from the drop down list, hit the recommend button and wait for the movie recommendations.")
        st.subheader("Model Performance Evaluation")
        st.markdown("Model evaluation aims to estimate the generalization accuracy of a model on future (unseen) data. Methods for evaluating a model's performance use a test set (i.e data not seen by the model) to evaluate model performance. This evaluation shows total efficiency as scores.")
        st.subheader("The Team:")
        st.markdown(" * Buhle Ntushelo ")
        st.markdown(" * Khanyisa Galela ")
        st.markdown(" * Keane Byrne ")
        st.markdown(" * Olwethu Mkhuhlane ")
        st.markdown(" * Londiwe cele ")


    if page_selection == "Solution Overview":
        st.title("Solution Overview")

        st.title("Content-based Filtering")
        st.image('resources/imgs/content_based.png', width=350)
        st.info("Content-based filtering, also referred to as cognitive filtering, recommends items based on a comparison between the content of the compared items, in this case its movie content, and a user profile. The content of each movie is represented as a set of descriptors or terms. Generally, Content-based filtering, makes recommendations based on user preferences for product features")

        st.title("Collaborative-based Filtering")
        st.image('resources/imgs/collab_based.jpg', width=350)
        st.info("Collaborative filtering filters information by using the interactions and data collected by the system from other users. It's based on the idea that people who agreed in their evaluation of certain items are likely to agree again in the future. Collaborative filtering basically mimics user-to-user recommendations.")

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
        st.image('resources/imgs/rating_distibution.PNG', width=400)
        st.image('resources/imgs/distribution_graph.PNG', width=400)
        st.info("""From this we can see that people are more likely to give a movie an * Average * rating more than a great or awful rating.
        This makes sense because most movies are average and not every movie is great or awful. This could also lead us to belive that there may be certain biasy around our data.""")
    
        st.image('resources/imgs/director_ratings.PNG', width=200)
        st.info("**These are the top 10 directors who have movies with the highest number of ratings Jonathan Nolan has the highest rating. this concludes that his movies are being watched and enjoyed by the users**")
        
        st.write("**Average of Movie Ratings per Year**")
        st.image('resources/imgs/rating_year.PNG', width=400)

        st.write("**Average Movie Rating per Week**")
        st.image('resources/imgs/ratings_week.PNG', width=400)

        st.write("**Number of ratings per year**")
        st.image('resources/imgs/number_ratings.PNG', width=400)

        st.write("**The Elbow Method**")
        st.image('resources/imgs/elbow.PNG', width=400)
        st.info("We use this visualization in order to obtain the optimal clusters for the data, we look for the * bend point * in the data, as we try to optimize the computational power while still having enough data to accuratly build a model from the data. ")

        st.write("**Dendrogram**")
        st.image('resources/imgs/dendogram.PNG', width=400)
        st.info("This is another way to visualize the data to obtain the optimal amount of clusters similar to the * Elbow Method *, we use this to pick the clusters that will maximize our efficientcy in the modelling process.")
if __name__ == '__main__':
    main()
