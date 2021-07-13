import numpy as np
import matplotlib.pyplot as plt


def reactions_pregnant_moderna_pfizer():
    """
    The two histograms visualizes the different reported symptoms from dose 1 and dose 2
    over a population of pregnant women. The first histogram visualizes the symptoms from
    the Pfizer/Biotech covid19 vaccine, and the second is for the Moderna covid19 vaccine.
    """

    N = 14  # Total number of paired charts

    # Pfizer
    # Counts
    dose_1 = np.array([7602, 2406, 1497, 795, 254, 256, 30, 492, 82, 209, 318, 117, 178, 20])
    dose_2 = np.array([5886,4231,3138,2916,1747,1648,315,1356,201,1267,411,316,277,18])
    # Percentages
    dose_1 = dose_1/np.sum(dose_1)
    dose_2 = dose_2/np.sum(dose_2)

    ind = np.arange(N) + 0.15  # The x locations for the groups
    width = 0.35  # The width of the bars
    xtra_space = 0.05 # Extra space between each pair of chart

    fig, (ax1, ax2) = plt.subplots(2)

    rects1 = ax1.bar(ind, dose_1, width, color="#2ca02c")
    rects2 = ax1.bar(ind + width + xtra_space, dose_2, width, color="#17becf")

    # add title
    ax1.set_title(
        "Reactons from vaccinated pregnant women with Pfizer/BioTech COVID-19", size=8
    )

    # Moderna
    # Counts
    dose_1 = np.array([7360, 2616, 1581, 1167, 442, 453, 62, 638, 77, 342, 739, 160, 189, 22])
    dose_2 = np.array([5388,4541,3662,3722,2755,2594,664,1909,357,1871,1051,401,332,18])
    # Percentages
    dose_1 = dose_1/np.sum(dose_1)
    dose_2 = dose_2/np.sum(dose_2)

    rects1 = ax2.bar(ind, dose_1, width, color="#2ca02c")
    rects2 = ax2.bar(ind + width + xtra_space, dose_2, width, color="#17becf")

    ax2.set_title(
        "Reactons from vaccinated pregnant women with Moderna COVID-19", size=8
    )

    # Adding some text for labels and axes ticks
    for axes in [ax1, ax2]:
        axes.set_ylabel("%")
        axes.legend(["Dose 1", "Dose 2"], fontsize=8)
        axes.set_xticks(ind + 0.15 + xtra_space)
        axes.set_xticklabels(
            (
                "Injection-site pain",
                "Fatigue",
                "Headache",
                "Myalgi",
                "Chills",
                "Fever/feverish",
                r"Temperature $\geq$38°C",
                "Nausea",
                "Vomiting",
                "Joint pain",
                "Injection-site swelling",
                "Abdonimal pain",
                "Diarrhea",
                "Rash",
            ),
            size=4,
        )

    plt.show()


def reactions_pregnant_vs_non_pregnant():
    """
    The following histograms visualizes the distribution of various symptoms after
    the first and second dose of the covid19 vaccine. The figures visualizes
    the difference between the various symptoms of pregnant vs. non-pregnant women.

    Each figure is seperated by different age groups of 16-24 years, 25-34 years,
    35-44 years, and 45-54 years.
    """

    N = 9  # Total number of paired charts

    ind = np.arange(N) + 0.35  # The x locations for the groups
    width = 0.35  # The width of the bars
    xtra_space = 0.05 # Extra space between each pair of chart

    fig, (ax1,ax2) = plt.subplots(2)

    # Pfizer
    # Counts
    dose_1_preg = np.array([315, 140, 118, 92, 35, 43, 49, 32, 44])
    dose_1_non_preg = np.array([77713,36837,28785,27871,12699,13296,10075,9687,14286])
    # Percentages
    dose_1_preg = dose_1_preg/np.sum(dose_1_preg)
    dose_1_non_preg = dose_1_non_preg/np.sum(dose_1_non_preg)


    rects1 = ax1.bar(ind, dose_1_preg, width, color="#2ca02c")
    rects2 = ax1.bar(ind + width + xtra_space, dose_1_non_preg, width, color="#17becf")

    # add some text for labels, title and axes ticks

    ax1.set_title(
        "Reactons from pregnant and non-pregant women of ages 16-24 years vaccinated with 1st dose against COVID-19",
        size=8,
    )

    # Moderna
    dose_2_preg = np.array([208 ,176 ,137 ,133 ,88 ,94 ,81 ,67 ,41]) # Counts
    dose_2_non_preg = np.array([41922,35801,32725,31945,24489,25297,14726,16804,9880]) # Counts

    dose_2_preg = dose_2_preg/np.sum(dose_2_preg) # Percentages
    dose_2_non_preg = dose_2_non_preg/np.sum(dose_2_non_preg) # Percentages

    rects1 = ax2.bar(ind, dose_2_preg, width, color="#2ca02c")
    rects2 = ax2.bar(ind + width + xtra_space, dose_2_non_preg, width, color="#17becf")

    ax2.set_title(
        "Reactons from pregnant and non-pregant women of ages 16-24 years vaccinated with 2nd dose against COVID-19",
        size=8,
    )


    for axes in [ax1, ax2]:
        axes.legend(["Pregnant", "Non-pregnant"], fontsize=8)
        axes.set_xticks(ind + 0.15 + xtra_space)
        axes.set_ylabel("%")
        axes.set_xticklabels(
            (
                "Injection-site pain",
                "Fatigue",
                "Headache",
                "Myalgi",
                "Chills",
                "Fever/feverish",
                "Nausea",
                "Joint pain",
                "Injection-site swelling",
            ),
            size=4,
        )

    plt.show()


if __name__ == "__main__":
    reactions_pregnant_moderna_pfizer()
    reactions_pregnant_vs_non_pregnant()
