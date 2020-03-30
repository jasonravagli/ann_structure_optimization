import matplotlib.pyplot as plt
import numpy as np

quasi_random_mean = 74.05
grid_mean = 73.83
pso_10_10_mean = 75.10
pso_border_mean = 74.57
pso_local_search_mean = 75.29

quasi_random_eval = 10
grid_eval = 250
pso_eval = 110
pso_border_eval = 110
pso_local_search_eval = 825

def generate_plots():
    names = ["Quasi-Random", "Grid Search", "PSO - Base", "PSO - Init. on Borders", "PSO - Local Search"]
    y_pos = [1, 2, 3, 4, 5]
    values = [quasi_random_mean, grid_mean, pso_10_10_mean, pso_border_mean, pso_local_search_mean]
    width = 0.5
    colors = ['#307533', '#C64933', '#394DA8', '#665AC4', '#5A93DD']

    # Table
    row_labels = ["Accuracy (%)", "N. Evaluations"]
    col_labels = names  # ["Grid Search", "PSO 10"]
    cell_text = [values, [quasi_random_eval, grid_eval, pso_eval, pso_border_eval, pso_local_search_eval]]  #[[grid_mean, pso_10_10_mean]]

    fig, ax = plt.subplots(1)
    plt.bar(names, values, width=width, color=colors, alpha=0.9, align='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    table = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='bottom')
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim(top=100)

    plt.show()


generate_plots()
