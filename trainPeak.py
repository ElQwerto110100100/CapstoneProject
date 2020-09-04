import csv

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
                print(epoch, acc)
                if float(acc) > float(peakAcc):
                    peakAcc = acc
                    peakEpoch = epoch
            except:
                pass
    print("peak accuracy: " + str(peakAcc) + " at epochs: " + str(peakEpoch))

findPeak(csvReader("Iris.csv"))