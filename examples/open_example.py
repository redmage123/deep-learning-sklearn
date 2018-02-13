#!/usr/bin/env python3

import csv


f = open('example_data.dat','r')

reader = csv.reader(f)


for line in reader:
    print (line)

f.close()
