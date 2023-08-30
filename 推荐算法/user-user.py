import math
import pandas as pd


def get_movies():
    data = pd.read_csv('movies.csv')  # 读取电影名和标签
    id = data['movieId']
    title = data['title']
    genres = data['genres']
    movies_title = {}  # 电影名字列表
    movies_genres = {}  # 电影类型列表
    i = 0
    for line in title:
        movies_title[id[i]] = line
        i += 1
    i = 0
    for line in genres:
        movies_genres[id[i]] = line.split("|")
        i += 1
    return movies_title, movies_genres


def get_rating():  # 读取train_set中的评分信息
    f = open('train_set.csv')
    r = f.readlines()
    r.pop(0)
    f.close()

    ratings = []  # 用户，电影，评分列表
    user_rate = {}  # 用户id为索引的电影评分list
    movie_user = {}  # 电影id为索引的用户list

    for line in r:
        rating = line.strip().split(',')
        ratings.append([int(rating[0]), int(rating[1]), float(rating[2])])

    for rating in ratings:

        if rating[0] in user_rate:
            user_rate[rating[0]].append((rating[1], rating[2]))
        else:
            user_rate[rating[0]] = [(rating[1], rating[2])]

        if rating[1] in movie_user:
            movie_user[rating[1]].append(rating[0])
        else:
            movie_user[rating[1]] = [rating[0]]

    return user_rate, movie_user


def pearson(user1, user2):
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    avg_x = 0.0
    avg_y = 0.0
    for key in user1:
        avg_x += key[1]
    avg_x /= len(user1)

    for key in user2:
        avg_y += key[1]
    avg_y /= len(user2)

    for key1 in user1:
        for key2 in user2:
            if key1[0] == key2[0]:
                sum_xy += (key1[1] - avg_x) * (key2[1] - avg_y)
                sum_y += (key2[1] - avg_y)**2
        sum_x += (key1[1] - avg_x)**2

    if sum_xy == 0.0:
        return 0
    sx_sy = math.sqrt(sum_x * sum_y)
    return sum_xy / sx_sy


def near_user(user_id, user_rate, movie_user):
    neighbors = []
    neighbors_dist = []
    for score in user_rate[user_id]:  # 每一部评过分的电影
        for neighbor in movie_user[score[0]]:  # 给该电影评过分的每个用户
            if neighbor != user_id and neighbor not in neighbors:
                neighbors.append(neighbor)
                neighbors_dist.append([pearson(user_rate[user_id], user_rate[neighbor]), neighbor])  # 计算相似度
    neighbors_dist.sort(reverse=True)
    return neighbors_dist


def predict_score(user_id, movie_id, user_rate, movie_user):
    neighbors_dist = near_user(user_id, user_rate, movie_user)
    sum = 0
    for score in user_rate[user_id]:
        sum += score[1]
    user_acc = sum / len(user_rate[user_id])  # 用户的评分均分
    sum1 = 0
    sum2 = 0
    for neighbor in neighbors_dist:
        if neighbor[0] < 0:
            break

        sum = 0
        sum_user = 0
        for score in user_rate[neighbor[1]]:  # 邻居对电影的评分列表
            sum += score[1]
        neighbor_acc = sum / len(user_rate[neighbor[1]])  # 邻居的评分平均分

        for score in user_rate[neighbor[1]]:
            if score[0] == movie_id:  # 找到对该电影的评分
                sum_user += neighbor[0]
                sum1 += neighbor[0] * (score[1] - neighbor_acc)  # 公式
        if sum_user == 0:  # 邻居未对该电影评分
            sum_user = neighbor[0]
        sum2 += sum_user
    pre_score = sum1 / sum2 + user_acc  # 公式
    return pre_score


def recommendation(user_id, user_rate, movie_user, k):
    neighbors_dist = near_user(user_id, user_rate, movie_user)[:k]
    movies = []
    sum = 0

    for score in user_rate[user_id]:  # 用户对电影的评分列表
        if score[0] not in movies:
            movies.append(score[0])
        sum += score[1]
    user_acc = sum / len(user_rate[user_id])  # 用户的评分均分

    recommend_dict = {}
    recommend_movie = {}
    for neighbor in neighbors_dist:
        if neighbor[0] < 0:
            break

        sum = 0
        for score in user_rate[neighbor[1]]:
            sum += score[1]
        neighbor_acc = sum / len(user_rate[neighbor[1]])

        for score in user_rate[neighbor[1]]:
            if score[0] not in movies:
                if score[0] not in recommend_dict:
                    recommend_movie[score[0]] = neighbor[0] * (score[1] - neighbor_acc)
                    recommend_dict[score[0]] = neighbor[0]
                else:
                    recommend_movie[score[0]] += neighbor[0] * (score[1] - neighbor_acc)
                    recommend_dict[score[0]] += neighbor[0]

    recommend_list = []
    for id in recommend_dict:
        recommend_list.append([recommend_movie[id] / recommend_dict[id] + user_acc, id])  # 公式
    recommend_list.sort(reverse=True)
    return recommend_list[:n], recommend_dict


if __name__ == '__main__':
    movies_title, movies_genres = get_movies()
    user_rate, movie_user = get_rating()

    id = int(input('用户id:'))
    if id != 0:
        k = int(input('k:'))
        n = int(input('n:'))
        recommend_list, recommend_dict = recommendation(id, user_rate, movie_user, k)

        print("{0:6}\t{1:6}\t{2:30}\t{3}".format('id', 'pre_score', 'title', 'tag'))
        for item in recommend_list:
            movie_id = item[1]
            print("{0:<6}\t{1:.6f}\t{2:30}\t{3}".format(movie_id, item[0], movies_title[movie_id], movies_genres[movie_id]))
    else:
        data = pd.read_csv('test_set.csv')
        userId = data['userId']
        movieId = data['movieId']
        rating = data['rating']
        k = 100
        sse = 0
        print('{0:4}\t{1:6}\t{2:6}\t{3:6}'.format('user_id', 'movie_id', 'pre_score', 'real_score'))
        for i in range(k):
            pre_score = predict_score(userId[i], movieId[i], user_rate, movie_user)
            print('{0:4}\t{1:6}\t{2:.6f}\t{3:.6f}'.format(userId[i], movieId[i], pre_score, rating[i]))
            sse += (pre_score - rating[i]) ** 2
        print("SSE=", sse)
