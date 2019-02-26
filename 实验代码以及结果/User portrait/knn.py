import csv
csv_file=csv.reader(open('cleaned.csv','r'))
print (csv_file)
for stu in csv_file:
      print(stu)
