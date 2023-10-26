import pandas as pd

class DemographicFiltering:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        
    def calculate_weighted_rating(self, n):
        qualified_movies = self.data[self.data['vote_count'].notnull() & self.data['vote_average'].notnull()]
        C = qualified_movies['vote_average'].mean()
        m = qualified_movies['vote_count'].quantile(0.8)
        qualified_movies = qualified_movies[(qualified_movies['vote_count'] >= m)]
        
        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (m + v) * C)

        qualified_movies['weighted_rating'] = qualified_movies.apply(weighted_rating, axis=1)
        top_movies = qualified_movies.sort_values('weighted_rating', ascending=False).head(n)
        return top_movies.loc[:,['title','release_date','weighted_rating']]
