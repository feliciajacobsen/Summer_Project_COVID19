import numpy as np
import matplotlib.pyplot as plt




def maternal_risk_vaccinated_unvaccinated():
    """
    The visualized data only include data collected from pregnant women.

    The following histograms visualises the distribution of various maternal
    risk factors associated with covid-19 for vaccinated women (against covid-19)
    and unvaccinated women.

    Params:
    -------
        None

    Returns:
    -------
        None
    """
    # Smokers
    vacc = (0)
    non_vacc = (10.5)
    ind = np.arange(1) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars
    fig, axs = plt.subplots(2,2, figsize=(7,7))
    rects1 = axs[0,0].bar(ind, vacc, width, color='#2ca02c')
    xtra_space = 0.05
    rects2 = axs[0,0].bar(ind + width + xtra_space, non_vacc, width, color='#17becf')
    # add some text for labels, title and axes ticks
    axs[0,0].set_ylabel("%")
    axs[0,0].set_title("Smokers", size=8)
    axs[0,0].legend(["Vaccinated","Unvaccinated"],prop={'size': 5})
    axs[0,0].set_xticks(ind+0.15+xtra_space)
    axs[0,0].set_xticklabels(("Smokers",), fontsize=8)

    # BMI groups
    vacc = (56.5,16.9,12.1,11.3,3.2)
    non_vacc = (39.6,28.3,15.6,9.6,6.8)
    ind = np.arange(5) + .15
    rects3 = axs[1,0].bar(ind, vacc, width, color='#2ca02c')
    xtra_space = 0.05
    rects4 = axs[1,0].bar(ind + width + xtra_space, non_vacc, width, color='#17becf')
    # add some text for labels, title and axes ticks
    axs[1,0].set_ylabel("%")
    axs[1,0].set_title("Pre-pregnancy BMI", size=8)
    axs[1,0].legend(["Vaccinated","Unvaccinated"],prop={'size': 5})
    axs[1,0].set_xticks(ind+0.15+xtra_space)
    axs[1,0].set_xticklabels(("<25","25-29","30-34","35-39","40+"), fontsize=8)


    # Gravidity
    vacc = (40,24.3,20.7,15)
    non_vacc = (29.3,27.9,18.8,24)
    ind = np.arange(4) + .15
    rects3 = axs[0,1].bar(ind, vacc, width, color='#2ca02c')
    xtra_space = 0.05
    rects4 = axs[0,1].bar(ind + width + xtra_space, non_vacc, width, color='#17becf')
    # add some text for labels, title and axes ticks
    axs[0,1].set_ylabel("%")
    axs[0,1].set_title("Gravidity", size=8)
    axs[0,1].legend(["Vaccinated","Unvaccinated"],prop={'size': 5})
    axs[0,1].set_xticks(ind+0.15+xtra_space)
    axs[0,1].set_xticklabels(("1","2","3","4+"), fontsize=8)


    # Education years
    vacc = (0,53.4,46.6)
    non_vacc = (4.7,81.9,13.4)
    ind = np.arange(3) + .15
    rects3 = axs[1,1].bar(ind, vacc, width, color='#2ca02c')
    xtra_space = 0.05
    rects4 = axs[1,1].bar(ind + width + xtra_space, non_vacc, width, color='#17becf')
    # add some text for labels, title and axes ticks
    axs[1,1].set_ylabel("%")
    axs[1,1].set_title("Education years", size=8)
    axs[1,1].legend(["Vaccinated","Unvaccinated"],prop={'size': 5})
    axs[1,1].set_xticks(ind+0.15+xtra_space)
    axs[1,1].set_xticklabels(("<12","12-16",">16"), fontsize=8)


    plt.show()



def risk_ratio_pregnant_women():
    """
    The following histograms visualizes the data of pregnant women infected with
    SARS-CoV-2. The risk ratio is taken between women with severe and non-severe
    disease.

    Params:
    -------
        None

    Returns:
    -------
        None

    """

    plt.figure(figsize=(8,8))
    values = [np.array([1.49]),np.array([1.56]),np.array([3.77]),np.array([0.93]),np.array([1.07]),np.array([1.73]),np.array([2.34]),np.array([2.96]),np.array([1.96]),np.array([2.81])]
    upper_cf = np.array([np.array([1.21]),np.array([1.09]),np.array([1.86]),np.array([0.55]),np.array([0.67]),np.array([0.78]),np.array([0.99]),np.array([1.17]),np.array([0.34]),np.array([1.67])])-values
    lower_cf = values-np.array([np.array([1.84]),np.array([2.13]),np.array([7.67]),np.array([1.56]),np.array([1.72]),np.array([3.85]),np.array([5.53]),np.array([7.47]),np.array([11.34]),np.array([4.75])])
    tot_cf = np.array([lower_cf, upper_cf])
    labels = ["Age >35 years","Obesity","Smoke","Nullparious","Multiparious","Chronic respiratory disease","Cardiac disease","Diabetes","Gestational diabetes mellitus","Preeclampsia(history/existing)"]

    w = .15 # With of each column
    n = len(values) # Number of bars to plot
    x = np.arange(0,1) # Center position of group on x axis


    for i, value in zip(range(10), values):
        position = x + (w*(1-n)/2) + i*w
        plt.bar(position, value, width=w, yerr=tot_cf[:,i], capsize=2)

    plt.xticks((x + (w*(1-n)/2) + np.arange(10)*w),labels,size=5,rotation=30)
    plt.ylabel("Risk ratio")
    plt.title("Risk ratios for various maternal factors of pregnant women infected with SARS-CoV-2 :\nSevere infection vs non-severe infection with 95% confidence interval.")
    plt.show()


if __name__ == "__main__":
    maternal_risk_vaccinated_unvaccinated()
    risk_ratio_pregnant_women()
