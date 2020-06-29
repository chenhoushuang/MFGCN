# -*- Encoding:UTF-8 -*-

import pandas as pd
import numpy as np

test_ratio = 0.2
np.random.seed(0)


class Data:
    def __init__(self, batch_size=128, name='ml-1m'):
        self.dataName = name
        self.batch_size = batch_size
        self.dataPath = "./data/" + self.dataName + "/"
        # Static Profile
        self.ftr_size = {}
        self.movie_id_map = None
        self.UserInfo = self.getUserInfo()
        self.MovieInfo = self.getMovieInfo()
        self.data = self.getData()
        self.n_train = 0
        self.n_test = 0
        self.train_set, self.test_set = self.train_test_split()
        self.norm_adj = self.get_norm_adj()

        del self.movie_id_map

    def getUserInfo(self):
        if self.dataName == "ml-1m":
            userInfoPath = self.dataPath + "users.dat"

            users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
            users = pd.read_table(userInfoPath, sep='::', header=None, names=users_title, engine='python')
            users = users.filter(regex='UserID|Gender|Age|JobID')
            users_orig = users.values

            # 将性别映射到0,1
            gender_map = {'F': 0, 'M': 1}
            users['Gender'] = users['Gender'].map(gender_map)
            # 将年龄组映射到0-6
            age_map = {val: idx for idx, val in enumerate(set(users['Age']))}
            users['Age'] = users['Age'].map(age_map)

            users['UserID'] -= 1
            self.ftr_size['Gender'] = 2
            self.ftr_size['Age'] = len(age_map)
            self.ftr_size['User'] = len(users)
            self.ftr_size['Job'] = len(set(users['JobID']))
            return users

    def getMovieInfo(self):
        if self.dataName == "ml-1m":
            movieInfoPath = self.dataPath + "movies.dat"

            movies_title = ['MovieID', 'Title', 'Genres']
            movies = pd.read_table(movieInfoPath, sep='::', header=None, names=movies_title, engine='python')
            movies = movies.filter(regex='MovieID|Genres')

            # 电影类型映射到0-18
            genres_set = set()
            for val in movies['Genres'].str.split('|'):
                genres_set.update(val)
            genres2int = {val: idx for idx, val in enumerate(genres_set)}
            genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in
                          enumerate(set(movies['Genres']))}
            movies['Genres'] = movies['Genres'].map(genres_map)

            # rearrange the move ids
            self.movie_id_map = {val: idx for idx, val in enumerate(movies['MovieID'])}
            movies['MovieID'] = movies['MovieID'].map(self.movie_id_map)

            self.ftr_size['Movie'] = len(movies)
            self.ftr_size['Genres'] = len(genres2int)
            return movies

    def getData(self):
        if self.dataName == "ml-1m":
            dataPath = self.dataPath + "ratings.dat"

            ratings_title = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
            ratings = pd.read_table(dataPath, sep='::', header=None, names=ratings_title, engine='python')

            ratings['UserID'] -= 1
            ratings['MovieID'] = ratings['MovieID'].map(self.movie_id_map)

            data = pd.merge(pd.merge(ratings, self.UserInfo), self.MovieInfo)
            data = data.sort_values(by=['TimeStamp'])

            return data

    def get_norm_adj(self):
        n_movies = self.ftr_size['Movie']
        n_users = self.ftr_size['User']
        norm_adj = np.zeros([n_users, n_movies], dtype=np.float32)
        # norm_adj[self.data['UserID'], self.data['MovieID']] = 1
        for k in self.train_set.keys():
            norm_adj[k][self.train_set[k]] = 1
        norm_adj = norm_adj / np.sqrt(norm_adj.sum(axis=1, keepdims=True))
        norm_adj = norm_adj.T
        # some items may not be clicked by any users
        index = np.where(norm_adj.sum(axis=1) != 0)
        norm_adj[index] = norm_adj[index] / np.sqrt(norm_adj.sum(axis=1, keepdims=True))[index]
        return norm_adj.T

    def train_test_split(self):
        sorted_data = self.data.sort_values(by=['UserID', 'MovieID'])
        UserID = sorted_data.UserID.values
        MovieID = sorted_data.MovieID.values
        cur_user = UserID[0]
        train_set = {}
        test_set = {}
        train_movies = []
        test_movies = []
        for i in range(len(UserID)):
            if UserID[i] == cur_user:
                if np.random.random() < test_ratio:
                    test_movies.append(MovieID[i])
                    self.n_test += 1
                else:
                    train_movies.append(MovieID[i])
                    self.n_train += 1
            else:
                if len(test_movies):
                    test_set[cur_user] = test_movies
                if len(train_movies):
                    train_set[cur_user] = train_movies
                cur_user = UserID[i]
                if np.random.random() < test_ratio:
                    test_movies = [MovieID[i]]
                    train_movies = []
                else:
                    test_movies = []
                    train_movies = [MovieID[i]]

        if len(test_movies):
            test_set[cur_user] = test_movies
        if len(train_movies):
            train_set[cur_user] = train_movies
        return train_set, test_set

    def sample_train(self):
        n_users = self.ftr_size['User']
        n_items = self.ftr_size['Movie']
        if self.batch_size <= n_users:
            users = np.random.choice(np.arange(n_users), self.batch_size, replace=False)
        else:
            users = np.arange(n_users)
        pos_items, neg_items = [], []
        for u in users:
            pos_items.append(np.random.choice(self.train_set[u], 1)[0])
            while True:
                neg_id = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_id not in self.train_set[u]:
                    neg_items.append(neg_id)
                    break
        return users, pos_items, neg_items

    def sample_test(self):
        test_users = list(self.test_set.keys())
        if self.batch_size <= len(test_users):
            users = np.random.choice(test_users, self.batch_size, replace=False)
        else:
            users = test_users

        pos_items, neg_items = [], []
        for u in users:
            pos_items.append(np.random.choice(self.test_set[u], 1)[0])
            while True:
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_set[u]):
                    neg_items.append(neg_id)
                    break
        return users, pos_items, neg_items


if __name__ == '__main__':
    data = Data()
    sorted_data = data.data.sort_values(by=['UserID', 'MovieID'])
    UserID = sorted_data.UserID.values
    MovieID = sorted_data.MovieID.values
    cur_user = UserID[0]
    cur_movies = []
    f = open('user_movie.txt', 'w')
    for i in range(len(UserID)):
        if UserID[i] == cur_user:
            cur_movies.append(MovieID[i])
        else:
            sorted(cur_movies)
            f.write("{} ".format(cur_user))
            f.write(cur_movies.__str__()[1:-1].replace(',', ' ') + '\n')
            cur_user = UserID[i]
            cur_movies = [MovieID[i]]
    f.write("{} ".format(cur_user))
    f.write(cur_movies.__str__()[1:-1].replace(',', ' '))
    f.close()
