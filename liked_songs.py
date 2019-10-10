import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

songs_df = pd.read_csv('data.csv')
songs_df.drop(columns = ["Unnamed: 0"], inplace = True)
songs_df.to_csv('data.csv')

# setting up output to display all rows and columns without condensing
pd.get_option('display.max_columns')
pd.set_option('display.max_rows', None)
print(songs_df.isnull().sum())

# all data types look correct
print(songs_df.dtypes)

# creating new df to remove the non-numerical columns. no need to create dummy col out of these as we are trying
num_songs_df = songs_df.drop(columns = ['song_title', 'artist'])

# heatmap shows loudness and energy are pos corr and loudness/energy are neg corr to accousticness. loudness with instrumentalness and duration_ms are the next most neg corr. 2nd most pos corr is valence (mood, like happy or sad) with danceability...maybe combine loudness and energy
# want to drop attributes that don't have much corr with any other attributes...key, mode, speechiness
correlation = num_songs_df.corr()
ax = sns.heatmap(correlation, center = 0)

# tried to create an interaction term, but it ruined model.
# num_songs_df['loudness_energy'] = num_songs_df['loudness'] * num_songs_df['energy']

# No real dupes. Probably seeing all time_signature vals are the same
print(songs_df[songs_df.duplicated() == True])

# Checking if the interaction column was added to df
# print(num_songs_df.head())


# prior to modeling with just numerical columns, looking to see if artist column plays a role in whether a song is liked or not
# if there is a trend where an artist is always either liked or not liked, then that could affect whether the user likes the song
# looks like some artists (i.e. beyonce, drake, future) includes a like and not liked, so it doesn't look like user always bases whether they like a song according to the artist. therefore, the other features most likely play a bigger role, so we don't have to dummify the artist column. that's good, as it could result it too many added columns. also, based on the artists of liked songs, it doesn't look like user sticks to one main genre.
print(songs_df[songs_df['artist'].duplicated()].sort_values(by = 'artist'))

# create a new df that doesn't include energy and loudness features since we created interaction term above
# model_songs_df = num_songs_df.drop(columns = ['loudness', 'energy', 'target'])

# train test split for modeling
X_train, X_test, y_train, y_test = train_test_split(num_songs_df, num_songs_df['target'])

# modeling
lr = LogisticRegression()

lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print(accuracy_score(y_test, pred))


plt.show()
