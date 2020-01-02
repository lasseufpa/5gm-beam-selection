# 5GM BEAM SELECTION 

Steps to run the code for ITA'2018 paper below. We are using Python 3.6.

1) Git clone this repository

2) Download datasets (for both classification and regression) avaliable at https://nextcloud.lasseufpa.org/s/MCwo2TdixCM7ryo and store the files in the folder datasets (for example: D:\github\5gm-beam-selection\datasets)

3) Go to folder regression (for example,  D:\github\5gm-beam-selection\regression) and execute:

python deep_convnet_regression.py

4) @TO-DO NEED to fix Go to folder classification (for example, D:\github\5gm-beam-selection\classification) and execute:

python deep_ann_classifier.py

For more information on creating the dataset and related tasks, see the Wiki page at https://github.com/lasseufpa/5gm-data/wiki

# Reference

If you use any data or code, please cite: "5G MIMO Data for Machine Learning: Application to Beam-Selection using Deep Learning", Aldebaro Klautau, Pedro Batista, Nuria Gonzalez-Prelcic, Yuyang Wang and Robert W. Heath Jr., ITA'2018 (available at http://ita.ucsd.edu/workshop/18/files/paper/paper_3313.pdf).
```
Bibtex entry:
@inproceedings{Klautau18,
  author    = {Aldebaro Klautau and Pedro Batista and Nuria Gonzalez-Prelcic and Yuyang Wang and Robert W. {Heath Jr.}},
  title     = {{5G} {MIMO} Data for Machine Learning: Application to Beam-Selection using Deep Learning},
  booktitle = {2018 Information Theory and Applications Workshop, San Diego},
  pages     = {1--1},
  year      = {2018},
  url       = {http://ita.ucsd.edu/workshop/18/files/paper/paper_3313.pdf}
}
```
