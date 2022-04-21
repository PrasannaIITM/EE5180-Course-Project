import numpy as np
import csv

for dataset in ["FCT", "Shrutime"]:
    
    models = ["cnn", "node", "tabnet", "dnfnet", "xgboost"]

    predMap = {}

    y_test = []

    accMap = {}

    file = open(f'Data/{dataset}/test.csv')
    csvreader = csv.reader(file)
    for i, row in enumerate(csvreader):
        if i > 0:
            if dataset == "FCT":
                y_test.append(int(row[-1]))
            elif dataset == "Shrutime":
                y_test.append(int(row[-3]))


    for model in models:
        file = open(f'Pred/{dataset}/{model}.csv')
        csvreader = csv.reader(file)

        preds = []

        if model in ["cnn", "dnfnet", "xgboost"]:
            for row in csvreader:
                row = list(map(float, row))
                preds.append(row)
            predMap[model] = np.array(preds)
        else:
            for i, row in enumerate(csvreader):
                if i > 0:
                    row = list(map(float, row))
                    preds.append(row[1:])
            predMap[model] = np.array(preds)

    for model in models:
        sum = 0
        for i in range(len(predMap[model])):
            x = np.argmax(predMap[model][i])

            if dataset == "FCT":
                y = y_test[i] - 1
            elif dataset == "Shrutime":
                y = y_test[i]

            sum += (x == y)
        accMap[model] = sum * 100/len(predMap[model])

    print(f"For the dataset {dataset}, accuracies of individual models : ", accMap)

    #create ensemble
    ensembleProbabs = predMap[models[0]]
    for i in range(1, len(models)):
        ensembleProbabs += predMap[models[i]]
    ensembleProbabs /= len(models)

    sum = 0
    for i in range(len(ensembleProbabs)):
        x = np.argmax(ensembleProbabs[i])

        if dataset == "FCT":
            y = y_test[i] - 1
        elif dataset == "Shrutime":
            y = y_test[i]

        sum += (x == y)
    ensembleAcc = sum * 100/len(ensembleProbabs)

    print(f"For the dataset {dataset}, accuracy after ensemble of all models = ", ensembleAcc)

