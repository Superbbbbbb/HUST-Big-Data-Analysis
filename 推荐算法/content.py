import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity


def get_movies():
    data = pd.read_csv("movies.csv")
    id = data['movieId']
    title = data['title']
    genres = data['genres']
    tags = []
    for line in genres:
        tags.append(line.split('|'))

    movie_id = []  # 电影id列表
    movie_info = {}  # 电影信息
    for i in range(len(id)):
        movie_info[id[i]] = [title[i], tags[i]]
        movie_id.append(int(id[i]))
    return movie_info, movie_id


def get_rating():  # 读取train_set中的评分信息
    f = open('train_set.csv')
    r = f.readlines()
    r.pop(0)
    f.close()

    ratings = []  # 用户，电影，评分列表
    user_rate = {}  # 用户id为索引的电影评分list
    user_movie = {}  # 电影id为索引的用户list

    for line in r:
        rating = line.strip().split(',')
        ratings.append([int(rating[0]), int(rating[1]), float(rating[2])])

    for rating in ratings:

        if rating[0] in user_rate:
            user_rate[rating[0]].append((rating[1], rating[2]))
        else:
            user_rate[rating[0]] = [(rating[1], rating[2])]

        if rating[0] in user_movie:
            user_movie[rating[0]].append(rating[1])
        else:
            user_movie[rating[0]] = [rating[1]]

    return user_rate, user_movie


def TF_IDF(movie_info):
    tags = []
    for i, item in movie_info.items():
        for tag in item[1]:
            if tag not in tags:
                tags.append(tag)

    movie_num = len(movie_info)
    tag_num = len(tags)
    tf_matrix = np.zeros([movie_num, tag_num])
    idf_matrix = np.zeros([movie_num, tag_num])
    tf_idf = np.zeros([movie_num, tag_num])

    a = 0
    for i, item in movie_info.items():
        for tag in item[1]:
            b = tags.index(tag)
            tf_matrix[a, b] = 1
            idf_matrix[a, b] = 1
        a = a + 1
    for i in range(movie_num):
        sum_of_row = sum(tf_matrix[i, :])  # 电影总数
        for j in range(tag_num):
            if tf_matrix[i, j]:
                tf_matrix[i, j] /= sum_of_row  # 词频=词在文件中出现次数/文件中词总数
    for i in range(tag_num):
        sum_of_col = sum(idf_matrix[:, i])  # 标签总数
        for j in range(movie_num):
            if idf_matrix[j, i]:  # IDF=log(文档总数/包含词的文档总数)
                idf_matrix[j, i] = math.log(movie_num / sum_of_col)
    for i in range(movie_num):
        for j in range(tag_num):
            tf_idf[i, j] = idf_matrix[i, j] * tf_matrix[i, j]  # TF-IDF=TD*IDF
    return tf_idf


def recommendation(user_id, movie_id, cos_sim, user_rate, user_movie):  # 求出每一部电影的预期评分并进行排序
    scores = user_rate[user_id]

    recommend_list = []
    recommend_dict = {}
    for i in range(len(movie_id)):
        if (movie_id[i]) not in user_movie[user_id]:
            sum = 0
            sum1 = 0
            sum2 = 0
            for score in scores:
                j = movie_id.index(score[0])
                if cos_sim[j, i] > 0:
                    if movie_id[i] in scores:
                        continue
                    else:
                        sum += score[1]
                        sum1 += cos_sim[j, i] * score[1]
                        sum2 += cos_sim[j, i]
                else:
                    continue
            if sum2 == 0:
                pre_score = sum / len(scores)  # 分母为0
            else:
                pre_score = sum1 / sum2  # 代入公式
            recommend_list.append([pre_score, movie_id[i]])
            recommend_dict[movie_id[i]] = pre_score
    recommend_list.sort(reverse=True)

    return recommend_list, recommend_dict


def predict_score(user_id, movie_id, user_rate, cos_sim, movieid):
    scores = user_rate[user_id]
    i = movie_id.index(movieid)
    sum = 0
    sum1 = 0
    sum2 = 0
    for score in scores:
        j = movie_id.index(score[0])
        if cos_sim[j, i] > 0:
            if movie_id[i] in scores:
                continue
            else:
                sum += score[1]
                sum1 += cos_sim[j, i] * score[1]
                sum2 += cos_sim[j, i]
        else:
            continue
    if sum2 == 0:
        pre_score = sum / len(scores)
    else:
        pre_score = sum1 / sum2
    return pre_score


if __name__ == "__main__":
    movies_info, movies_id = get_movies()
    user_rate, user_movie = get_rating()
    cos_sim = cosine_similarity(TF_IDF(movies_info))

    user_id = int(input('用户id:'))
    if user_id != 0:
        n = int(input('n:'))
        recommend_list, recommend_dict = recommendation(user_id, movies_id, cos_sim, user_rate, user_movie)
        print("{0:6}\t{1:6}\t{2:30}\t{3}".format('id', 'pre_score', 'title', 'tag'))
        for i in range(n):
            j = recommend_list[i][1]
            print('{0:6}\t{1:.6f}\t{2:30}\t{3}'.format(j, recommend_list[i][0], movies_info[j][0],
                                                       movies_info[j][1]))
    else:
        test_data = pd.read_csv("test_set.csv")
        userId = test_data['userId']
        movieId = test_data['movieId']
        rating = test_data['rating']
        k = 100
        sse = 0
        print('{0:4}\t{1:6}\t{2:6}\t{3:6}'.format('user_id', 'movie_id', 'pre_score', 'real_score'))
        for i in range(k):
            pre_score = predict_score(userId[i], movies_id, user_rate, cos_sim, movieId[i])
            print('{0:4}\t{1:6}\t{2:.6f}\t{3:.6f}'.format(userId[i], movieId[i], pre_score, rating[i]))
            sse += (pre_score - rating[i]) ** 2
        print("SSE=", sse)
