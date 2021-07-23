import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


"""
These functions visualizes the outcomes of women with coronavirus.
The following data is collected from international studies.
"""


def overall_hist():
    N = 3 # total number of paired charts

     # Counts per 1000 cases
    preg = (10.5,2.9,1.5) # pregnant women
    non_preg = (3.9,1.1,1.2) # non-pregnant women

    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35  # the width of the bars
    xtra_space = 0.05 # the extra space between each pair of charts

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in pregnant women\n with Sars-Cov-2 vs non-pregnant women with Sars-Cov-2")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation', 'Maternal death'))

    plt.show()


def age_25_34_hist():
    """
    The following histogram visualizes the distibution of various outcomes
    associated with covid-19 infection between pregnant and non-pregnant women
    in the age group of 25 to 34 years of age.
    """
    N = 2 # total number of paired charts

    preg = (9.1, 2.3)
    non_preg = (3.5, 0.9)

    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35  # the width of the bars
    xtra_space = 0.05 # the extra space between each pair of charts

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')

    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 of ages 25-34:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation'))

    plt.show()



def age_35_44_hist():
    """
    The following histogram visualizes the distibution of various outcomes
    associated with covid-19 infection between pregnant and non-pregnant women
    in the age group of 35 to 44 years of age.
    """
    N = 3 # total number of paired charts

    preg = (19.4, 6.5, 4.2)
    non_preg = (6.4, 1.8, 2.3)


    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35  # the width of the bars
    xtra_space = 0.05 # the extra space between each pair of charts

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 of ages 35-44:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation', 'Maternal death'))

    plt.show()



def diabetes_hist():
    """
    The following histogram visualizes the distibution of various outcomes
    associated with covid-19 infection between pregnant and non-pregnant women
    with underlying diabetes.
    """
    N = 3 # total number of paired charts

    preg = (58.5, 23.4,14.1)
    non_preg = (44.8, 16.0, 12.7)

    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35  # the width of the bars
    xtra_space = 0.05 # the extra space between each pair of charts

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 and underlying diabetes:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation', 'Maternal death'))

    plt.show()



def CVD_hist():
    """
    The following histogram visualizes the distibution of various outcomes
    associated with covid-19 infection between pregnant and non-pregnant women
    with cardivascular disease (CVD)
    """
    N = 3 # total number of paired charts

    preg = (42.8, 10.7, 23.0)
    non_preg = (32.1, 10.6, 11.6)

    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35  # the width of the bars
    xtra_space = 0.05 # the extra space between each pair of charts

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, preg, width, color='#2ca02c')
    rects2 = ax.bar(ind + width + xtra_space, non_preg, width, color='#17becf')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Count per 1000 cases")
    ax.set_title("Outcomes in women with Sars-Cov-2 and underlying CVD:\n pregnant vs non-pregnant")
    ax.legend(["Pregnant","Non-pregnant"])


    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(('ICU admissions', 'Invasive ventilation','Maternal Death'))

    plt.show()


def outcomes_pregnant_vs_nonpregnant_histogram():
    """
    The following histogram visualises the number of events per 1000 cases of
    outcomes associated with covid-19 infection between pregnant and non-pregnant.

    Each group of histogram is associated with a specific outcome, where each bin
    in a group represent an age group or a group with underlying diabetes or CVD.
    """
    plt.subplot(211)
    outcomes = ["ICU Admission", "Invasive Ventilation","Maternal Death"]
    # each array in the list represent a collection of each population group for each of the outcomes
    values = [np.array([10.5, 2.9, 1.5]), np.array([9.1, 2.3, 1.2]), np.array([19.4, 6.5, 4.2]), np.array([58.5,23.4,14.1]), np.array([42.8,19.7,23.0])]
    n = len(values)                # Number of bars to plot
    w = 0.15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["Overall", "Age 25-34", "Age 35-44", "Underlying diabetes", "Underlying CVD"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes, size=8)
    plt.ylabel('Count per 1000 cases')
    plt.title("Outcomes in pregant women with SARS-Cov-2", size=8)
    plt.legend(fontsize=8)

    plt.subplot(212)
    # each array in the list represent a collection of each population group for each of the outcomes
    values = [np.array([3.9, 1.1, 1.2]), np.array([3.5, 0.9, 0.9]), np.array([6.4,1.8,2.3]), np.array([44.8,16.0,12.7]), np.array([32.1,10.6,11.6])]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label)

    plt.xticks(x, outcomes, size=8)
    plt.ylabel('Count per 1000 cases')
    plt.title("Outcomes in non-pregant women with SARS-Cov-2", size=8)
    plt.legend(fontsize=8)

    plt.show()



def RR():
    """
    The following histogram visualises the risk ratios of various outcomes between
    pregnant women and non-pregnant women. Both groups are infected with SARS-CoV-2.
    """

    outcomes = ["ICU Admission", "Invasive Ventilation","Maternal Death"]
    values = [np.array([3.0, 2.9, 1.7]), np.array([2.4, 2.5, 1.2]), np.array([3.2,3.6,2.0]), np.array([1.5,1.7,1.5]), np.array([1.5,1.9,2.2])]
    # 95% confidence interval
    upper_cf = np.array([np.array([3.4,3.8,2.4]),np.array([3.0,3.7,2.1]),np.array([4.0,5.4,3.2]),np.array([2.2,3.3,3.5]),np.array([2.6,4.5,4.8])])-values
    lower_cf = values-np.array([np.array([2.6,2.2,1.2]),np.array([2.0,1.6,0.7]), np.array([2.5,2.4,1.2]),np.array([1.0,0.9,0.6]),np.array([0.9,0.8,1.0])])
    tot_cf = np.array([lower_cf, upper_cf])
    n = len(values)                # Number of bars to plot
    w = .15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["Overall", "Age 25-34", "Age 35-44", "Underlying diabetes", "Underlying CVD", "95% confidence"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label, yerr=tot_cf[:,i], capsize=2)
    plt.xticks(x, outcomes)
    plt.ylabel("Risk ratio")
    plt.title("Risk ratios for various outcomes of women with Sars-Cov-2:\n pregnant vs non-pregnant with 95% confidence interval.")
    plt.legend(fontsize=8)

    plt.show()



def OR():
    """
    The following histogram visualizes the odds ratio of various maternal risk factors
    associated with severe SARS-CoV-2 infection.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    outcomes = ["Severe disease","ICU Admission", "Invasive Ventilation","Maternal Death"]
    values = [np.array([1.83, 2.11, 1.72,0.91]), np.array([2.37,2.71,6.61,2.27]), np.array([1.81,1.70,5.26,2.53]), np.array([2.0,4.72,68.82,4.25]), np.array([2.12,4.67,18.61,14.88])]
    # 95% confidence interval
    upper_cf = np.array([np.array([2.63,2.63,4.97,3.72]),np.array([3.07,6.63,22.02,4.31]), np.array([2.20,2.15,15.68,8.17]),np.array([3.48,9.41,420.48,9.95]),np.array([2.78,11.22,1324.16,52.81])])-values
    lower_cf = values-np.array([np.array([1.27,1.69,0.60,0.22]),np.array([1.83,1.10,1.98,1.20]),np.array([1.49,1.34,1.76,0.78]),np.array([1.14,2.37,9.69,1.82]),np.array([1.62,1.94,0.26,4.19])])
    tot_cf = np.array([lower_cf, upper_cf])
    labels_cf = np.array([["1.27-2.63","1.69-2.63","0.60-4.97","0.22-3.72"], ["1.83-3.07","1.10-6.63","1.98-22.02","1.20-4.31"], ["1.49-2.20","1.34-2.15","1.76-15.68","0.78-8.17"], ["1.14-3.48","2.37-9.41","9.69-420.48","1.82-9.95"], ["1.62-2.78","1.94-11.22","0.26-1324.16","4.19-52.81"]])
    n = len(values)                # Number of bars to plot
    w = .15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = [r"Age $\geq$ 35", r"BMI $\geq$ 30", "Any Comorbidity", "Chronic hypertension", "Pre-existing diabetes"]
    for i, value, label in zip(range(5), values, labels):
        position = x + (w*(1-n)/2) + i*w
        bars=ax.bar(position, value, width=w, label=label)
        ax.bar_label(container=bars,labels=labels_cf[i], padding=-5, size=5, rotation=45)

    plt.xticks(x, outcomes)
    plt.ylabel("Odds ratio")
    plt.title("Odds ratios of maternal risk factors assiciated with severe SARS-Cov-2")
    plt.legend(fontsize=8)

    plt.show()



def preg_women_hist():
    """
    The following histogram visualizes the odds ratio of different outcomes between
    pregnant women with and without SARS-CoV-2 infection.
    """
    plt.subplot(211)
    outcomes = ["Maternal outcomes"]
    values = [np.array([18.58]), np.array([1.47]), np.array([2.85])]
    upper_cf = np.array([np.array([45.82]),np.array([1.91]),np.array([7.52])])-values
    lower_cf = values-np.array([np.array([7.53]),np.array([1.14]),np.array([1.08])])
    tot_cf = np.array([lower_cf, upper_cf])
    n = len(values)                # Number of bars to plot
    w = .15                        # With of each column
    x = np.arange(0, len(outcomes))   # Center position of group on x axis
    labels = ["ICU admission", "Preterm birth <37 weeks", "All cause mortality"]

    for i, value, label in zip(range(3), values, labels):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, label=label, yerr=tot_cf[:,i], capsize=2)

    plt.xticks(x, outcomes)
    plt.ylabel("Odds ratio")
    plt.title("Odds ratios for various outcomes of pregnant women:\n SARS-Cov-2 infected vs non-infected with 95% confidence interval.")
    plt.xlim([-0.5,0.5])
    plt.legend()

    plt.subplot(212)
    outcomes = ["Perinatal outcomes"]
    values = (2.84)
    cf_tot = np.array([values-np.array([1.25]), np.array([6.45])-values])
    plt.bar(0, values, width=0.15, label="Stillbirth", yerr=cf_tot, capsize=2)
    plt.xticks(np.arange(0, len(outcomes)), outcomes)
    plt.ylabel("Odds ratio")
    plt.xlim([-0.5,0.5])
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #overall_hist()
    #age_25_34_hist()
    #age_35_44_hist()
    #diabetes_hist()
    #CVD_hist()
    outcomes_pregnant_vs_nonpregnant_histogram()
    RR()
    OR()
    preg_women_hist()
