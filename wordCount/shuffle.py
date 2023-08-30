import threading
import time


def shuffle(readfile):
    read = open(readfile)
    write1 = open('shuffle1', 'a')
    write2 = open('shuffle2', 'a')
    write3 = open('shuffle3', 'a')
    for line in read:
        line = line.strip()
        word, count = line.split(',', 1)
        if 'd' <= word[0] <= 'o':
            write2.write("{},{}\n".format(word, count))
        elif 'p' <= word[0] <= 'z':
            write3.write("{},{}\n".format(word, count))
        else:
            write1.write("{},{}\n".format(word, count))


if __name__ == '__main__':
    t1 = threading.Thread(target=shuffle('combine1'), args=("t1",))
    t2 = threading.Thread(target=shuffle('combine2'), args=("t2",))
    t3 = threading.Thread(target=shuffle('combine3'), args=("t3",))
    t4 = threading.Thread(target=shuffle('combine4'), args=("t4",))
    t5 = threading.Thread(target=shuffle('combine5'), args=("t5",))
    t6 = threading.Thread(target=shuffle('combine6'), args=("t6",))
    t7 = threading.Thread(target=shuffle('combine7'), args=("t7",))
    t8 = threading.Thread(target=shuffle('combine8'), args=("t8",))
    t9 = threading.Thread(target=shuffle('combine9'), args=("t9",))
    start = time.perf_counter()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()

    t1.join()
    print("t1: %s s" % (time.perf_counter() - start))
    t2.join()
    print("t2: %s s" % (time.perf_counter() - start))
    t3.join()
    print("t3: %s s" % (time.perf_counter() - start))
    t4.join()
    print("t4: %s s" % (time.perf_counter() - start))
    t5.join()
    print("t5: %s s" % (time.perf_counter() - start))
    t6.join()
    print("t6: %s s" % (time.perf_counter() - start))
    t7.join()
    print("t7: %s s" % (time.perf_counter() - start))
    t8.join()
    print("t8: %s s" % (time.perf_counter() - start))
    t9.join()
    print("t9: %s s" % (time.perf_counter() - start))
