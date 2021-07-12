import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


"""
These functions visualizes the outcomes of women with coronavirus.
The following data is collected from international studies.
"""


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


def age_25_34_hist():

    N = 2

    preg = (9.1, 2.3)
    non_preg = (3.5, 0.9)


    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 of ages 25-34:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation'))

    plt.show()



def age_35_44_hist():

    N = 3

    preg = (19.4, 6.5, 4.2)
    non_preg = (6.4, 1.8, 2.3)


    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 of ages 35-44:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation', 'Maternal Death'))

    plt.show()



def diabetes_hist():
    N = 3

    preg = (58.5, 23.4,14.1)
    non_preg = (44.8, 16.0, 12.7)


    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 and underlying diabetes:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Maternal Death'))

    plt.show()



def CVD_hist():
    N = 3

    preg = (42.8, 10.7, 23.0)
    non_preg = (32.1, 10.6, 11.6)

    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 and underlying CVD:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Maternal Death'))

    plt.show()


def outcomes_pregnant_histogram():
    """
    The following histogram visualises the number of events per 1000 cases for
    pregnant women infected with SARS-CoV-2.

    The histogram visualises the number of events for each of the various outcomes of
    the disease.
    """
    outcomes = ["ICU Admission", "Invasive Ventilation","Maternal Death"]
    # each array in the list represent a collection of each population group for each of the outcomes
    values = [np.array([10.5, 2.9, 1.5]), np.array([9.1, 2.3, 1.2]), np.array([19.4, 6.5, 4.2]), np.array([58.5,0,14.1]), np.array([42.8,19.7,23.0])]
    n = len(values)                # Number of bars to plot
    w = 0.15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["Overall", "Age 25-34", "Age 35-44", "Underlying diabetes", "Underlying CVD"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes);
    plt.ylabel('Count per 1000 cases')
    plt.title("Outcomes in pregant women with Sars-Cov-2")
    plt.legend()

    plt.show()



def outcomes_non_pregnant_histogram():
    outcomes = ["ICU Admission", "Invasive Ventilation","Maternal Death"]
    # each array in the list represent a collection of each population group for each of the outcomes
    values = [np.array([3.9, 1.1, 1.2]), np.array([3.5, 0.9, 0.9]), np.array([6.4,1.8,2.3]), np.array([44.8,16.0,12.7]), np.array([32.1,10.6,11.6])]
    n = len(values) # Number of bars to plot
    w = 0.15  # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["Overall", "Age 25-34", "Age 35-44", "Underlying diabetes", "Underlying CVD"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes)
    plt.ylabel('Count per 1000 cases')
    plt.title("Outcomes in non-pregant women with Sars-Cov-2")
    plt.legend()

    plt.show()



def RR():
    """
    The following histogram visualises the risk ratios of various outcomes in
    pregnant women infected with SARS-CoV-2 and non-pregnant women with SARS-CoV-2.
    """

    outcomes = ["ICU Admission", "Invasive Ventilation","Maternal Death"]
    values = [np.array([3.0, 2.9, 1.7]), np.array([2.4, 2.5, 1.2]), np.array([3.2,3.6,2.0]), np.array([1.5,1.7,1.5]), np.array([1.5,1.9,2.2])]
    n = len(values)                # Number of bars to plot
    w = .15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["Overall", "Age 25-34", "Age 35-44", "Underlying diabetes", "Underlying CVD"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes)
    plt.ylabel("Risk ratio")
    plt.title("Risk ratios for various outcomes of women with Sars-Cov-2:\n pregnant vs non-pregnant with significant ratios.")
    plt.legend()

    plt.show()



def OR():
    """
    The following histogram visualizes the odds ratio of maternal risk factors
    associated with severe SARS-CoV-2.
    """

    outcomes = ["Severe disease","ICU Admission", "Invasive Ventilation","Maternal Death"]
    values = [np.array([1.83, 2.11, 1.72,0.91]), np.array([2.37,2.71,6.61,2.27]), np.array([1.81,1.70,5.26,2.53]), np.array([2.0,4.72,68.82,4.25]), np.array([2.12,4.67,14.88,18.61])]
    n = len(values)                # Number of bars to plot
    w = .15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = [r"Age $\geq$ 35", r"BMI $\geq$ 30", "Any Comorbidity", "Chronic hypertension", "Pre-existing diabetes"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes)
    plt.ylabel("Odds ratio")
    plt.title("Odds ratios of maternal risk factors assiciated with severe Sars-Cov-2")
    plt.legend()

    plt.show()



def preg_women_hist():
    """
    The following histogram visualizes the odds ratio of different outcomes between
    pregnant women with and without Sars-CoV-2 infection.
    """

    outcomes = ["Maternal outcomes","Perinatal outcomes"]
    values = [np.array([18.58,0]), np.array([1.47,0]), np.array([2.85,0]), np.array([0, 2.84])]
    n = len(values)                # Number of bars to plot
    w = .15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["ICU admission", "Preterm birth <37 weeks", "All cause mortality", "Stillbirth"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes)
    plt.ylabel("Odds ratio")
    plt.title("Odds ratios for various outcomes of pregnant women:\n Sars-Cov-2 infected vs non-infected with significant ratios.")
    plt.legend()

    plt.show()



if __name__ == "__main__":
    overall_hist()
    age_25_34_hist()
    age_35_44_hist()
    diabetes_hist()
    CVD_hist()
    outcomes_pregnant_histogram()
    outcomes_non_pregnant_histogram()
    RR()
    OR()
    preg_women_hist()
