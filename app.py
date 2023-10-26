import streamlit as st
from streamlit import cache_data

from demographic_filtering import DemographicFiltering
from content_filtering import ContentFiltering

def main():

    st.set_page_config(
        page_title="Movie Recommender TMDB",
        page_icon=":)",
        )

    cache_data.clear()  # Clear Streamlit cache
    st.title("Movie Recommender System")
    
    filtering_method = st.sidebar.selectbox(
        "Select Filtering Method",
        ("Demographic Filtering", "Content-Based Filtering")
    )
    
    data_file = 'tmdb_5000_movies.csv'
    
    if filtering_method == "Demographic Filtering":
        n = st.sidebar.number_input("Number of Top Movies", min_value=1, max_value=100, value=10)
        demographic_filter = DemographicFiltering(data_file)
        demographic_top_movies = demographic_filter.calculate_weighted_rating(n)
        demographic_top_movies = demographic_top_movies.reset_index(drop=True)
        st.subheader(f"Top {n} Movies by Demographic Filtering")
        st.write(demographic_top_movies)

    else:
        content_filter = ContentFiltering(data_file)
        movie_title = None
        while not movie_title:
            movie_title = st.text_input("Enter a movie title:")
        
        n = st.sidebar.number_input("Number of Recommended Movies", min_value=1, max_value=10, value=5)
        
        if st.button("Get Content-Based Recommendations"):
            recommended_movies = content_filter.content_recommender(movie_title, n)
            recommended_movies = recommended_movies.reset_index(drop=True)
            if recommended_movies.empty:
                st.warning("No recommendations found for the given movie title.")
            else:
                st.subheader(f"Top {n} Recommended Movies Based on '{movie_title}'")
                st.write(recommended_movies)

if __name__ == "__main__":
    main()

