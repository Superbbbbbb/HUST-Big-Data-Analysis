import numpy as np

if __name__ == '__main__':
    f = open('sent_receive.csv')
    edges = [line.strip('\n').split(',') for line in f]
    edges.pop(0)

    nodes = []
    for edge in edges:
        if edge[1] not in nodes:
            nodes.append(edge[1])
        if edge[2] not in nodes:
            nodes.append(edge[2])

    n = len(nodes)
    M = np.zeros([n, n])
    for edge in edges:
        from_ = nodes.index(edge[1])
        to = nodes.index(edge[2])
        M[to, from_] = 1

    for j in range(n):
        s = sum(M[:, j])
        for i in range(n):
            if M[i, j]:
                M[i, j] /= s

    pagerank = np.ones(n) / n  # 初始化
    e = 300000  # 误差
    b = 0.85

    while e > 0.00000001:
        next = b * np.dot(M, pagerank) + (1 - b) * np.ones(n) / n  # 迭代
        next /= sum(next)
        e = max(map(abs, next - pagerank))  # 求误差
        pagerank = next

    print('id\tpagerank')
    for i in range(n):
        print('%s\t%f' % (nodes[i], pagerank[i]))
