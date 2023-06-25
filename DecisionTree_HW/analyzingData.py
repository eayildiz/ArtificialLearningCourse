import pandas as pd
import numpy as np

result_file = open("result_file.txt", "w")
dataset = pd.read_csv('Iris.csv')

result_file.write("\tDATASET SUMMARY\n\n")
result_file.write(dataset.head().to_string())
result_file.write("\n\nShape is: " + str(dataset.shape))
#result_file.write(dataset.info())
result_file.write("\n\nNull values of table\n" + dataset.isnull().sum().to_string() + "\n\n")
result_file.write("General info of dataset\n" + dataset.describe().to_string())

result_file.write("\n\n\nDATASET CORRELATION MATRIX WITH ALL COLUMNS\n\n")
corr_matrix = dataset[dataset.columns[1:]].corr()['Species'][:]
result_file.write(dataset.corr().to_string() + "\n\n")
result_file.write("\t\tSpecies\n" + corr_matrix.to_string())


result_file.close()