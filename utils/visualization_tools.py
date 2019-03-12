import numpy as np
from matplotlib import pyplot as plt

def _flatten_list(list_):
    flatten_list = []
    for i in list_:
        for j in i:
            flatten_list.append(j)
    return np.array(flatten_list)

def plot_generations(history):
    accuracies = history

    points = [[j+1 for i in range(len(accuracies[j]))] for j in range(accuracies.shape[0])]
    points = _flatten_list(points)
    colors = np.random.rand(points.shape[0])
    flatten_accuracies = _flatten_list(accuracies)
    plt.scatter(points, flatten_accuracies, c=colors, alpha=0.2)
#     regr = linear_model.LinearRegression()
#     regr.fit(points.reshape(-1,1), flatten_accuracies)
#     pred_accuracies = regr.predict(points.reshape(-1,1))       
#     plt.plot(points,pred_accuracies)
    # plt.xticks(list(range(1,19)))
    plt.xlabel("Generation")
    plt.ylabel("Individual Fitness")

def plot_average_generations(history):
    accuracies = np.mean(history,0).reshape(history.shape[1])

    points = np.arange(len(accuracies))
    plt.plot(points, accuracies)

    plt.xlabel("Generation")
    plt.ylabel("Individual Average Fitness")