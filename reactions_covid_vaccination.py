import numpy as np
import matplotlib.pyplot as plt


def overall_hist():

    N = 3

    preg = (10.5,2.9,1.5)
    non_preg = (3.9,1.1,1.2)

    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in pregnant women\n with Sars-Cov-2 vs non-pregnant women with Sars-Cov-2")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation', 'Maternal death'))

    plt.show()
