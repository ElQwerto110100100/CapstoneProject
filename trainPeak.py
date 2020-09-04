import csv
import os

def csvReader(fname):
    with open(fname, newline='') as csvfile:
        #spread out the data
        datasetReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = [row for row in datasetReader]
    return data

def findPeak(data):
    peakAcc = 0.0
    peakEpoch = 0 
    for item in data:
        for epoch, acc in enumerate(item):
            
            try:
                if float(acc) > float(peakAcc):
                    peakAcc = acc
                    peakEpoch = epoch
            except:
                pass
    print("peak accuracy: " + str(peakAcc) + " at epochs: " + str(peakEpoch))

print("------------------ Start Analysis ------------------ ")

path = r'C:\Users\joshy\Desktop\Github\CapstoneProject\data'
for subdir, dirs, files in os.walk(path):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".csv"):
            print (filepath.strip(path))
            findPeak(csvReader(filepath))

print("------------------ End Analysis ------------------ ")