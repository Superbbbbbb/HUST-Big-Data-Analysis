import pandas as pd
import numpy as np


def get_data():
    data = pd.read_csv("Groceries.csv")
    # 读取列数据
    items = data['items']
    data = np.array(items)
    # 将列数据转化为多维数组
    list = []
    for line in data:
        line = line.strip('{').strip('}').split(',')
        s = []
        for i in line:
            s.append(i)
        list.append(s)
    return list


def is_apriori(ck_item, Lk):
    # 任何非频繁的(k-1)项集都不是频繁k项集的子集，因此Ck+1中每一个集合的子集都应该在Lk中
    for item in ck_item:
        sub_item = ck_item - frozenset([item])
        if sub_item not in Lk:
            return False
    return True


def get_C1(data):
    C1 = set()
    for items in data:
        for item in items:
            C1.add(frozenset([item]))
    return C1


def get_Ck(Lk, k):  # 通过合并Lk-1中前k-2项相同的项来获得Ck中的项
    C = set()
    len_Lk = len(Lk)
    list_Lk = list(Lk)
    for i in range(len_Lk):
        for j in range(i + 1, len_Lk):
            l1 = list(list_Lk[i])[0:k - 2]
            l2 = list(list_Lk[j])[0:k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                Ck_item = list_Lk[i] | list_Lk[j]
                if is_apriori(Ck_item, Lk):  # 舍弃非频繁项集
                    C.add(Ck_item)
    return C


def get_Lk(data_set, Ck, min_support, support_data):
    L = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    data_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / data_num) >= min_support:
            L.add(item)
            support_data[item] = item_count[item] / data_num
    return L


def get_rule(L, support_data, min_confidence):
    rule_list = []
    sub_set_list = []
    for Lk in L:
        for frequent_set in Lk:
            for sub_set in sub_set_list:
                if sub_set.issubset(frequent_set):  # 寻找上一轮循环中出现的frequent_set的子集
                    conf = support_data[frequent_set] / support_data[sub_set]
                    rule = (sub_set, frequent_set - sub_set, conf)
                    if conf >= min_confidence and rule not in rule_list:
                        rule_list.append(rule)
            sub_set_list.append(frequent_set)
    return rule_list


if __name__ == "__main__":
    data = get_data()
    min_support = 0.005
    min_confidence = 0.5
    support_data = {}
    C1 = get_C1(data)
    L1 = get_Lk(data, C1, min_support, support_data)
    print('L1:%d' % (len(L1)))
    L = [L1]
    Lk = L1.copy()

    for k in range(2, 4):
        Ck = get_Ck(Lk, k)
        Lk = get_Lk(data, Ck, min_support, support_data)
        print('L%d:%d' % (k, len(Lk)))
        Lk = Lk.copy()
        L.append(Lk)

    rule_list = get_rule(L, support_data, min_confidence)
    print('频繁项集总数：%d' % (len(support_data)))
    print(support_data)
    print('关联规则总数:%d' % (len(rule_list)))
    for rule in rule_list:
        print(rule)

    with open('L1.csv', 'w') as f:
        for key in L[0]:
            f.write('{},\t{}\n'.format(key, support_data[key]))
    with open('L2.csv', 'w') as f:
        for key in L[1]:
            f.write('{},\t{}\n'.format(key, support_data[key]))
    with open('L3.csv', 'w') as f:
        for key in L[2]:
            f.write('{},\t{}\n'.format(key, support_data[key]))
    with open('rule.csv', 'w') as f:
        for item in rule_list:
            f.write('{}\t{}\t{}\t: {}\n'.format(item[0], "of", item[1], item[2]))
