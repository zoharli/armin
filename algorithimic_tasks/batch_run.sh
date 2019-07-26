#! /bin/bash

python3 train.py -task=copy -model=lstm	  -lstm_size=300		>>lstm_copy.txt
python3 train.py -task=copy -model=ntm	  -lstm_size=120		>>ntm_copy.txt
python3 train.py -task=copy -model=dnc    -lstm_size=120		>>dnc_copy.txt
python3 train.py -task=copy -model=sam    -lstm_size=120		>>sam_copy.txt
python3 train.py -task=copy -model=tardis -lstm_size=100        >>tardis_copy.txt
python3 train.py -task=copy -model=awta   -lstm_size=100		>>awta_copy.txt
python3 train.py -task=copy -model=armin  -lstm_size=100		>>armin_copy.txt

python3 train.py -task=repeat -model=lstm	-lstm_size=300		>>lstm_repeat.txt
python3 train.py -task=repeat -model=ntm	-lstm_size=120		>>ntm_repeat.txt
python3 train.py -task=repeat -model=dnc    -lstm_size=120		>>dnc_repeat.txt
python3 train.py -task=repeat -model=sam    -lstm_size=120		>>sam_repeat.txt
python3 train.py -task=repeat -model=tardis -lstm_size=100      >>tardis_repeat.txt
python3 train.py -task=repeat -model=awta   -lstm_size=100		>>awta_repeat.txt
python3 train.py -task=repeat -model=armin  -lstm_size=100		>>armin_repeat.txt

python3 train.py -task=recall -model=lstm	-lstm_size=300		>>lstm_recall.txt
python3 train.py -task=recall -model=ntm	-lstm_size=120		>>ntm_recall.txt
python3 train.py -task=recall -model=dnc    -lstm_size=120		>>dnc_recall.txt
python3 train.py -task=recall -model=sam    -lstm_size=120		>>sam_recall.txt
python3 train.py -task=recall -model=tardis -lstm_size=100      >>tardis_recall.txt
python3 train.py -task=recall -model=awta   -lstm_size=100		>>awta_recall.txt
python3 train.py -task=recall -model=armin  -lstm_size=100		>>armin_recall.txt

python3 train.py -task=sort -model=lstm	  -lstm_size=300		>>lstm_sort.txt
python3 train.py -task=sort -model=ntm	  -lstm_size=120		>>ntm_sort.txt
python3 train.py -task=sort -model=dnc    -lstm_size=120		>>dnc_sort.txt
python3 train.py -task=sort -model=sam    -lstm_size=120		>>sam_sort.txt
python3 train.py -task=sort -model=tardis -lstm_size=100        >>tardis_sort.txt
python3 train.py -task=sort -model=awta   -lstm_size=100		>>awta_sort.txt
python3 train.py -task=sort -model=armin  -lstm_size=100		>>armin_sort.txt
