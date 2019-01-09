import pandas as pd
import os
import numpy as np
from collections import Counter


def load_profile(profile_name):
    return pd.read_json('instagram-profilecrawl/profiles/'+profile_name)

def get_profile_names(base_path = 'instagram-profilecrawl/profiles'):
    list_profiles = os.listdir(base_path)
    return [profile for profile in list_profiles if '.json' in profile]

def extract_profile_data(df):
    username, num_of_posts, followers, following = df[['username', 'num_of_posts', 'followers', 'following']].values[0]
    return username, num_of_posts, followers, following

def extract_post_data(df):
    comments = []
    mentions = []
    for info_post in df['posts']:
        dict_infos_post = dict(info_post)
        comments.append(dict_infos_post['comments']['count'])

        print(dict_infos_post)
        mentions.extend(dict_infos_post['mentions'])

    most_commun_comments = [comm[0] for comm in Counter(mentions).most_common(3)]
    return np.mean(comments), np.std(comments), len(comments), most_commun_comments


def main():

    list_profiles = get_profile_names()
    for profile in list_profiles:
        df = load_profile(profile)
        profile_data = extract_profile_data(df)
        print(extract_post_data(df))



if __name__ =='__main__':
    main()