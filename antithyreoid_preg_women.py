import numpy as np
import matplotlib.pyplot as plt




def antithyreoid_pregnant_women():
    """
    The visualized data only include data collected of pregnant women.

    The following histograms visualises the distribution between the number
    of vaccinated and the no. of unvaccinated for different maternal risk factors
    associated with covid-19 vaccination.
    """
    # Smokers
    risk_preeclampsia = (4.7, 0.66, 0.46)
    risk_malformation = (1.2, 1.2, 1.2)
    ind = np.arange(3) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind,risk_preeclampsia, width, color='#2ca02c')
    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space, risk_malformation, width, color='#17becf')
    # add some text for labels, title and axes ticks
    ax.set_ylabel("Odds ratio", size=8)
    ax.set_title("Weighing up the potential and harms of antithyreoid drugs treatment of different types of hyperthyroidism", size=8)
    ax.legend(["Risk of preeclampsia due to\n untreated hyperthyrodism","Risk of any congenital malformation\n due to ATD treatment in pregnancy"],prop={'size': 5})
    ax.set_xticks(ind+0.15+xtra_space)
    ax.set_xticklabels(("Overt hyperthyroidism","Subclinical hyperthyrodism","Gestational hyperthyrodism",), fontsize=8)


    plt.show()



if __name__ == "__main__":
    antithyreoid_pregnant_women()
