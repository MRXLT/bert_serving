# -*- coding:utf-8 -*-
from time import time
import sys
import multiprocessing
from bert_service import BertService

#t_num = sys.argv[1]
batch_size = 1
data_list = []
with open("./check/data-e.txt") as f:
    for line in f.readlines():
        data_list.append([line.strip()])
print(len(data_list))
sys.stdout.flush()

start = time()


def process(batch_size, turn, check=False, profile=True, max_seq_len=128):
    if check:
        check_list = []
        with open("./check/check-cased.txt") as f:
            for line in f.readlines():
                line = line.strip().split(" ")[5:]
                check_list.append(line)
        print(len(check_list[0]))

    bc = BertService(
        profile=profile,
        emb_size=768,
        model_name='bert_uncased_L-12_H-768_A-12',
        do_lower_case=True,
        max_seq_len=max_seq_len)
    bc.connect('127.0.0.1', 8010)
    p_start = time()
    total_time = 0
    op_time = 0
    infer_time = 0
    request_time = 0
    max_diff = 0
    copy_time = 0
    if profile:
        for i in range(turn):
            re_time = bc.encode(data_list[i:i + batch_size])
            total_time += re_time[0]
            request_time += re_time[1]
            op_time += re_time[2]
            infer_time += re_time[3]
            copy_time += re_time[4]
            if i == 0:
                print("first time cost:" + str(total_time))
        p_end = time()

        print("batch_size:" + str(batch_size) + " " + str(i + 1) +
              " query cost " + str(p_end - p_start) + "s" + " total_time:" +
              str(total_time) + "s" + " request_time:" + str(request_time) +
              "s" + " copy_time:" + str(copy_time) + "s" + " op_time:" + str(
                  op_time) + "ms" + " infer_time:" + str(infer_time) + "ms")
    elif check:
        for i in range(turn):
            result = bc.encode(data_list[i:i + batch_size])
            for k in range(batch_size):
                for j in range(768):
                    diff = float(result[k][j]) - float(check_list[i + k][j])
                    if abs(diff) > max_diff:
                        max_diff = abs(diff)
                    if abs(diff) > 0.01:
                        print([result[k][j], check_list[i + k][j]])
                        print(data_list[i])
                        return -1

        print(max_diff)
    sys.stdout.flush()


for i in [1]:
    process(i, 1000, profile=True, check=False, max_seq_len=128)
