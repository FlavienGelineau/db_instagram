import csv

import pandas as pd
import os
import numpy as np
from collections import Counter

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import LinearRegression

nltk.download('vader_lexicon')

def load_profile(profile_name):
    return pd.read_json('instagram-profilecrawl/profiles/'+profile_name)

def get_profile_names(base_path = 'instagram-profilecrawl/profiles'):
    list_profiles = os.listdir(base_path)
    return [profile for profile in list_profiles if '.json' in profile]

def extract_profile_data(df):
    username, num_of_posts, followers, following = df[['username', 'num_of_posts', 'followers', 'following']].values[0]
    return {'username':username,
            'num_of_posts' :num_of_posts,
            'followers' : followers,
            'following' : following
            }

def extract_post_data(df, n_most_used_tags = 3, n_last_likes = 5):
    comments = []
    mentions = []
    sentiments = {
        'neg' : [],
        'pos' : [],
        'neu' : [],
        'compound' :[]
    }
    all_likes = []
    last_likes = []
    sid = SentimentIntensityAnalyzer()

    for info_post in df['posts']:
        dict_infos_post = dict(info_post)
        comments.append(dict_infos_post['comments']['count'])
        mentions.extend(dict_infos_post['mentions'])
        likes = dict_infos_post['likes']['count']
        if len(last_likes)<n_last_likes:
            last_likes.append(likes)
        all_likes.append(likes)

        sents = sid.polarity_scores(dict_infos_post['caption'])
        for name_sent, value in sents.items():
            sentiments[name_sent].append(value)
    most_commun_comments = [comm[0] for comm in Counter(mentions).most_common(n_most_used_tags)]
    means = [np.mean(val) for _, val in sentiments.items()]
    growth_likes = LinearRegression().fit(np.array([i for i in range(len(all_likes))]).reshape(-1, 1),
                                          np.array(all_likes).reshape(-1,1)).coef_[0][0]

    pre_result = {'n_mean_comments':np.mean(comments),
            'std_n_comments' : np.std(comments),
            'n comments':len(comments),
            'mean_neg':means[0],
            'mean_pos':means[1],
            'mean_neu':means[2],
            'mean_compound':means[3],
            'mean_likes' : np.mean(likes),
            'growth_likes' : growth_likes
                  }

    for i, last_like in enumerate(last_likes):
       pre_result['n_likes_last_picture_{}'.format(i)] = last_likes[i]
    for i, most_commun_comment in enumerate(most_commun_comments):
        pre_result['most_commun_comments_{}'.format(i)] = most_commun_comment
    return pre_result


def main(sub_file = 'res.csv'):
    all_dicts = []

    list_profiles = get_profile_names()
    for profile in list_profiles:
        df = load_profile(profile)
        profile_data = extract_profile_data(df)
        post_data = extract_post_data(df)
        post_data.update(profile_data)
        all_dicts.append(post_data)

    result = {}
    for i, dict_profile in enumerate(all_dicts):
        for name, value in dict_profile.items():
            if i==0:
                result[name] = [value]
            else:
                result[name].append(value)
    pd.DataFrame(result).to_csv(sub_file, index=False)

if __name__ =='__main__':
    main()