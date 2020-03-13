import matplotlib.pyplot as plt
import numpy as np

grid_mean = 0.7208
pso_10_10_mean = 0.7225

grid_combinations = 250
pso_10_10_combinations = 110

def generate_plots():
    names = ["Grid Search", "PSO 10"]
    y_pos = [1, 2]
    values = [grid_mean, pso_10_10_mean]
    width = 0.5
    colors = ['red', 'blue']

    # Table
    row_labels = ["Accuracy"]
    col_labels = ["Grid Search", "PSO 10"]
    cell_text = [[grid_mean, pso_10_10_mean]]



    # objects = ('bar1', 'bar2')
    # w = 0.3
    # y_pos = (w/2., w*1.5)
    # performance = [grid_combinations, pso_10_10_combinations]
    # stds = [0.3, 0.5]
    # plt.bar(y_pos, performance, width=w, align='center', yerr=stds, capsize=5, alpha=0.5, color=colors)
    # plt.gca().set_xlim([-1.,1.5])
    # plt.xticks(y_pos, names)
    # plt.ylabel('Time (seconds)')
    #
    # plt.table(cellText=[["Grid Search", str(grid_combinations)], ["PSO 10", str(pso_10_10_combinations)]],
    #                   rowLabels=["Grid Search", "PSO 10"],
    #                   colLabels=["Accuracy", "Accuracy"],
    #                   loc='bottom')
    #
    # plt.show()

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(1, grid_mean, width, color='g')
    # rects2 = ax.bar(2, pso_10_10_mean, width, color='#acc8f0')
    #
    # plt.figure()
    # plt.bar(names, values, width, color=colors)
    # plt.suptitle('Accuracy Comparison')
    # plt.show()
    #
    # values = [grid_combinations, pso_10_10_combinations]
    #
    # plt.figure()
    # plt.bar(names, values, width, color=colors)
    # plt.suptitle('Combinations Comparison')
    # plt.show()

    fig, ax = plt.subplots(1)
    plt.bar(names, values, width=width, color=colors, alpha=0.5, align='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    table = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='bottom')
    table.scale(1, 1.5)
    plt.ylabel('Accuracy')

    plt.show()

