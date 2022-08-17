#from codecs import ignore_errors
import csv
import os
train = {}
dev = {}

with open('train.txt', 'r', encoding='cp850', errors='') as tr:
    #data.decode("utf-8")
    lines = tr.read().lower().splitlines()
    #lines = lines
tr.close()


train_set = lines[:700000]
valid_set = lines[700000:]

#print(lines[:8])
with open('train_text.tsv', 'wt', newline='', encoding='cp850') as out_file1:
    with open('train_label.tsv', 'wt', newline='', encoding='cp850') as label_train:
        tsv_writer = csv.writer(out_file1)
        tsv_writer2 = csv.writer(label_train)
        #tsv_writer.writerow(['sentence', 'label'])
        for line in train_set:
            parts = line.split('\t', 1)
            #sen = parts[1]
            tsv_writer.writerow(parts[0])
            tsv_writer.writerow(parts[1])     
            tsv_writer2.writerow("1")   
            tsv_writer2.writerow("0")   
out_file1.close()

with open('valid_text.tsv', 'wt', newline='', encoding='cp850') as out_file2:
    with open('valid_label.tsv', 'wt', newline='', encoding='cp850') as label_valid:
        tsv_writer = csv.writer(out_file2)
        tsv_writer2 = csv.writer(label_valid)
        #tsv_writer.writerow(['sentence', 'label'])
        for line in valid_set:
            parts = line.split('\t', 1)
            #sen = parts[1]
            tsv_writer.writerow(parts[0])
            tsv_writer.writerow(parts[1])     
            tsv_writer2.writerow("1")   
            tsv_writer2.writerow("0")   
out_file2.close()