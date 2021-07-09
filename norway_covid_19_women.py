import numpy as np
import matplotlib.pyplot as plt



def cases_vs_vaccinated_hist():
    """
    The following histograms visualises the distribution
    between the number of vaccinated women in norway and a separate histogram
    for distribution of infected women in different age groups.
    """
    # Smokers
    vaccinated = (8052,55733,46864,75181)

    ind = np.arange(4) # the x locations for the groups
    width = 0.3       # the width of the bars
    fig, (ax1, ax2) = plt.subplots(2)
    rects2 = ax1.bar(ind + width, vaccinated, width, color='#1f77b4')
    # add some text for labels, title and axes ticks
    ax1.set_ylabel("Count per 100 000")
    ax1.set_title("Number of vaccinated norwegian women of different age groups", size=10)
    ax1.set_xticks(ind+width)
    ax1.set_xticklabels(("16-17","18-24","25-39","40-44"),fontsize=8)

    # BMI groups
    cases = (3494,3664,2740,2650)

    rects4 = ax2.bar(ind + width, cases, width, color='#1f77b4')
    # add some text for labels, title and axes ticks
    ax2.set_ylabel("Count per 100 000")
    ax2.set_title("Number of norwegian women with covid-19 of different age groups", size=10)
    ax2.set_xticks(ind+width)
    ax2.set_xticklabels(("10-19","20-29","30-39","40-49"), fontsize=8)


    plt.show()



if __name__ == "__main__":
    cases_vs_vaccinated_hist()
