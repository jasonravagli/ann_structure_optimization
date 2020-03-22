import matplotlib.pyplot as plt
import numpy as np

quasi_random_mean = 0.7405
grid_mean = 0.7383
pso_10_10_mean = 0.7510
pso_border_mean = 0.7457
pso_local_search_mean = 0.7330

grid_combinations = 250
pso_10_10_combinations = 110

def generate_plots():
    names = ["Quasi Random", "Grid Search", "PSO"]
    y_pos = [1, 2, 3]
    values = [quasi_random_mean, grid_mean, pso_10_10_mean]
    width = 0.5
    colors = ['#307533', '#C64933', '#394DA8']

    # Table
    row_labels = ["Accuracy"]
    col_labels = names  # ["Grid Search", "PSO 10"]
    cell_text = [values]  #[[grid_mean, pso_10_10_mean]]

    fig, ax = plt.subplots(1)
    plt.bar(names, values, width=width, color=colors, alpha=1, align='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    table = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='bottom')
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim(top=1)

    plt.show()


    names = ["Base", "Init. on Borders", "Local Search"]
    y_pos = [1, 2, 3]
    values = [pso_10_10_mean, pso_border_mean, pso_local_search_mean]
    width = 0.5
    colors = ['#394DA8', '#665AC4', '#5A93DD']

    # Table
    row_labels = ["Accuracy"]
    col_labels = names  # ["Grid Search", "PSO 10"]
    cell_text = [values]  #[[grid_mean, pso_10_10_mean]]

    fig, ax = plt.subplots(1)
    plt.bar(names, values, width=width, color=colors, alpha=1, align='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    table = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='bottom')
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim(top=1)

    plt.show()


generate_plots()
