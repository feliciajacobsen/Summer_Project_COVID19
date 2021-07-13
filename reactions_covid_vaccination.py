import numpy as np
import matplotlib.pyplot as plt


def reactions_pregnant_hist():

    N = 14

    # Pfizer
    dose_1 = (7602, 2406, 1497, 795, 254, 256, 30, 492, 82, 209, 318, 117, 178, 20)
    dose_2 = (5886,4231,3138,2916,1747,1648,315,1356,201,1267,411,316,277,18)

    ind = np.arange(N) + 0.35  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    rects1 = ax1.bar(ind, dose_1, width, color="#2ca02c")

    xtra_space = 0.05
    rects2 = ax1.bar(ind + width + xtra_space, dose_2, width, color="#17becf")

    # add some text for labels, title and axes ticks
    ax1.set_ylabel("Count")
    ax1.set_title(
        "Reactons from vaccinated pregnant women with Pfizer/BioTech COVID-19", size=8
    )
    ax1.legend(["Dose 1", "Dose 2"], fontsize=8)

    #Moderna
    dose_1 = (7360,2616 ,1581 ,1167 ,442,453,62,638,77,342,739,160,189,22)
    dose_2 = (5388 ,4541 ,3662 ,3722 ,2755 ,2594 ,664 ,1909 ,357,1871 ,1051 ,401,332,18)

    rects1 = ax2.bar(ind, dose_1, width, color="#2ca02c")
    rects2 = ax2.bar(ind + width + xtra_space, dose_2, width, color="#17becf")

    ax2.set_ylabel("Count")
    ax2.set_title(
        "Reactons from vaccinated pregnant women with Moderna COVID-19", size=8
    )
    ax2.legend(["Dose 1", "Dose 2"], fontsize=8)
    ax2.set_xticks(ind + 0.15 + xtra_space)
    ax2.set_xticklabels(
        (
            "Injection-site pain",
            "Fatigue",
            "Headache",
            "Myalgi",
            "Chills",
            "Fever",
            r"Temperature $\geq$38°C",
            "Nausea",
            "Vomiting",
            "Joint pain",
            "Injection-site swelling",
            "Abdonimal pain",
            "Diarrhea",
            "Rash",
        ), size=4
    )

    plt.show()


if __name__ == "__main__":
    reactions_pregnant_hist()
