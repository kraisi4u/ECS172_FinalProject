import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import random

artists = pd.read_csv('artists.dat', sep='\t', usecols=['id', 'name'])
tags = pd.read_csv('tags.dat', sep='\t', encoding='latin-1', usecols=['tagID', 'tagValue'])
user_artists = pd.read_csv('user_artists.dat', sep='\t')
user_friends = pd.read_csv('user_friends.dat', sep='\t')
user_taggedartists = pd.read_csv('user_taggedartists.dat', sep='\t')

sns.set_theme(style="whitegrid")

# top 10 artists by total plays
artist_weights = user_artists.groupby('artistID')['weight'].sum().reset_index()
artist_weights = artist_weights.merge(artists, left_on='artistID', right_on='id')
top_10_artists = artist_weights.sort_values(by='weight', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_artists, x='weight', y='name', palette='viridis')
plt.title('Top 10 Artists (by Total Plays)')
plt.xlabel('Number of Listens')
plt.ylabel('Artist')
plt.tight_layout()

plt.savefig('top10artistsPlays.png')
plt.close()

# top 10 artists by unique listeners
artist_listeners = user_artists.groupby('artistID').size().reset_index(name='listen_count')
artist_listeners = artist_listeners.merge(artists, left_on='artistID', right_on='id')
top_10_listeners = artist_listeners.sort_values(by='listen_count', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_listeners, x='listen_count', y='name', palette='plasma')
plt.title('Top 10 Artists (by Total Unique Listeners)')
plt.xlabel('Number of Unique Listeners')
plt.ylabel('Artist')
plt.tight_layout()

plt.savefig('top10artistsUniqueListeners.png')
plt.close()

# top 10 most popular tags
tag_counts = user_taggedartists.groupby('tagID').size().reset_index(name='count')
tag_counts = tag_counts.merge(tags, on='tagID')
top_10_tags = tag_counts.sort_values(by='count', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_tags, x='count', y='tagValue', palette='magma')
plt.title('Top 10 Tags')
plt.xlabel('Number of Times Tagged')
plt.ylabel('Tag')
plt.tight_layout()

plt.savefig('top10tags.png')
plt.close()

# number of shared artists between friends
user_artists_dict = user_artists.groupby('userID')['artistID'].apply(set).to_dict()
friend_pairs = list(zip(user_friends['userID'], user_friends['friendID']))

shared_artists_count = [len(user_artists_dict.get(u1, set()).intersection(user_artists_dict.get(u2, set()))) for u1, u2 in friend_pairs]

plt.figure(figsize=(10, 5))
sns.histplot(shared_artists_count, bins=40, color='mediumseagreen', kde=True)
plt.title('How Many Artists Do Friends Actually Share?')
plt.xlabel('Number of Shared Artists')
plt.ylabel('Frequency (Pairs of Friends)')
plt.xlim(0, 30) # 0 to 30 shared artists
plt.tight_layout()

plt.savefig('sharedArtists.png')
plt.close()

# temporal data
timestamps_df = pd.read_csv('user_taggedartists-timestamps.dat', sep='\t')

# clean any bad/corrupted dates (was getting 1956 as a possible time, which is impossible)
# fix dates from UNIX
timestamps_df['datetime'] = pd.to_datetime(timestamps_df['timestamp'], unit='ms')
clean_df = timestamps_df[timestamps_df['datetime'].dt.year >= 2005].copy()
clean_df['year_month'] = clean_df['datetime'].dt.to_period('M')

monthly_activity = clean_df.groupby('year_month').size().reset_index(name='event_count')
monthly_activity['date'] = monthly_activity['year_month'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_activity, x='date', y='event_count', color='crimson', linewidth=2.5)
plt.fill_between(monthly_activity['date'], monthly_activity['event_count'], color='crimson', alpha=0.3)
plt.title('Timeline of User Tagging Activity (2005 - 2011)')
plt.xlabel('Year')
plt.ylabel('Number of Tagging Events')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(bottom=0)
ax.set_xlim(monthly_activity['date'].min(), monthly_activity['date'].max())

plt.tight_layout()

plt.savefig('activityTimeline.png')
plt.close()
