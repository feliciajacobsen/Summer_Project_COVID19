import numpy as np
import matplotlib.pyplot as plt


def reactions_pregnant_moderna_pfizer():
    """
    The two histograms visualizes the different reported symptoms from dose 1 and dose 2
    over a population of pregnant women. The first histogram visualizes the symptoms from
    the Pfizer/Biotech covid19 vaccine, and the second is for the Moderna covid19 vaccine.
    """

    N = 14  # Total number of paired charts

    ind = np.arange(N) + 0.15  # The x locations for the groups
    width = 0.35  # The width of the bars
    xtra_space = 0.05  # Extra space between each pair of chart

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))

    # Pfizer
    # Percentages
    dose_1 = np.array(
        [7602, 2406, 1497, 795, 254, 256, 30, 492, 82, 209, 117, 178, 20, 318]
    )/9052.
    dose_2 = np.array(
        [5886, 4231, 3138, 2916, 1747, 1648, 315, 1356, 201, 1267, 316, 277, 18, 411]
    )/6638.


    rects1 = ax1.bar(ind, dose_1, width, color="#2ca02c")
    rects2 = ax1.bar(ind + width + xtra_space, dose_2, width, color="#17becf")

    # add title
    ax1.set_title(
        "Reactons from vaccinated pregnant women with Pfizer/BioTech COVID-19", size=8
    )

    # Moderna
    # Percentages
    dose_1 = np.array(
        [7360, 2616, 1581, 1167, 442, 453, 62, 638, 77, 342, 160, 189, 22, 739]
    )/7930.
    dose_2 = np.array(
        [5388, 4541, 3662, 3722, 2755, 2594, 664, 1909, 357, 1871, 401, 332, 18, 1051]
    )/5635.


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
                "Abdonimal pain",
                "Diarrhea",
                "Rash",
                "Injection-site swelling",
            ),
            size=5,
        )

    plt.show()


def data_agebands():
    """
    Data over vaccinated pregnant and pregnant women with 1st and 2nd dose against
    COVID19 of agebands 16-24 years, 25-34 years, 35-44 years, and 45-54 years.
    """
    dose_1_preg_16_24 = np.array([315, 140, 118, 92, 35, 43, 49, 32, 44])
    dose_1_non_preg_16_24 = np.array(
        [77713, 36837, 28785, 27871, 12699, 13296, 10075, 9687, 14286]
    )
    dose_2_preg_16_24 = np.array([208, 176, 137, 133, 88, 94, 81, 67, 41])
    dose_2_non_preg_16_24 = np.array(
        [41922, 35801, 32725, 31945, 24489, 25297, 14726, 16804, 988]
    )

    dose_1_preg_25_34 = np.array([9729, 3270, 1973, 1239, 415, 404, 726, 325, 659])
    dose_1_non_preg_25_34 = np.array(
        [235590, 99834, 74571, 67552, 29549, 30430, 22885, 24757, 36749]
    )
    dose_2_preg_25_34 = np.array([7409, 5829, 4555, 4405, 2958, 2808, 2186, 2059, 928])
    dose_2_non_preg_25_34 = np.array(
        [155824, 131341, 114395, 117239, 88204, 87749, 47459, 65663, 34862]
    )

    dose_1_preg_35_44 = np.array([4635, 1514, 900, 558, 207, 224, 334, 159, 306])
    dose_1_non_preg_35_44 = np.array(
        [272408, 101279, 78344, 64512, 2823, 2835, 2302, 2618, 40269]
    )
    dose_2_preg_35_44 = np.array([3416, 2597, 1947, 1950, 1339, 1230, 938, 903, 434])
    dose_2_non_preg_35_44 = np.array(
        [186293, 150261, 127174, 133039, 96430, 93154, 49561, 77283, 42650]
    )

    dose_1_preg_45_54 = np.array([283, 98, 87, 73, 39, 38, 21, 35, 48])
    dose_1_non_preg_45_54 = np.array(
        [253077, 85171, 71531, 55860, 26149, 25656, 19265, 24962, 39135]
    )
    dose_2_preg_45_54 = np.array([241, 170, 161, 150, 117, 110, 60, 109, 59])
    dose_2_non_preg_45_54 = np.array(
        [164883, 125443, 108313, 109791, 79522, 76909, 38210, 66102, 39962]
    )

    data_array = np.array((
        dose_1_preg_16_24,
        dose_1_non_preg_16_24,
        dose_2_preg_16_24,
        dose_2_non_preg_16_24,
        dose_1_preg_25_34,
        dose_1_non_preg_25_34,
        dose_2_preg_25_34,
        dose_2_non_preg_25_34,
        dose_1_preg_35_44,
        dose_1_non_preg_35_44,
        dose_2_preg_35_44,
        dose_2_non_preg_35_44,
        dose_1_preg_45_54,
        dose_1_non_preg_45_54,
        dose_2_preg_45_54,
        dose_2_non_preg_45_54))

    """
    Divide into each age band into numpy arrays containing pregnant women with dose 1,
    non-pregnant with dose 1, pregnant with dose 2, and non-pregnant with dose 2.
    """
    age_band_1 = np.array(
        [
            data_array[0] / np.sum(data_array[0]),
            data_array[1] / np.sum(data_array[1]),
            data_array[2] / np.sum(data_array[2]),
            data_array[3] / np.sum(data_array[3]),
        ]
    )
    age_band_2 = np.array(
        [
            data_array[4] / np.sum(data_array[4]),
            data_array[5] / np.sum(data_array[5]),
            data_array[6] / np.sum(data_array[6]),
            data_array[7] / np.sum(data_array[7]),
        ]
    )
    age_band_3 = np.array(
        [
            data_array[8] / np.sum(data_array[8]),
            data_array[9] / np.sum(data_array[9]),
            data_array[10] / np.sum(data_array[10]),
            data_array[11] / np.sum(data_array[11]),
        ]
    )
    age_band_4 = np.array(
        [
            data_array[12] / np.sum(data_array[12]),
            data_array[13] / np.sum(data_array[13]),
            data_array[14] / np.sum(data_array[14]),
            data_array[15] / np.sum(data_array[15]),
        ]
    )

    return (age_band_1, age_band_2, age_band_3, age_band_4)


def reactions_pregnant_vs_non_pregnant():
    """
    The following histograms visualizes the distribution of various symptoms after
    the first and second dose of the covid19 vaccine. The figures visualizes
    the difference between the various symptoms of pregnant vs. non-pregnant women.

    Each figure is seperated by different age bands 1-4.

    Age band 1 = 16-24 years
    Age band 2 = 25-34 years,
    Age band 3 = 35-44 years,
    Age band 4 = 45-54 years.
    """

    age_band_1, age_band_2, age_band_3, age_band_4 = data_agebands()

    N = 9  # Total number of paired charts

    ind = np.arange(N) + 0.35  # The x locations for the groups
    width = 0.35  # The width of the bars
    xtra_space = 0.05  # Extra space between each pair of chart

    age_band_title = ["16-24", "25-34", "35-44", "45-54"]

    for i, title, age_band in zip(
        range(4), age_band_title, [age_band_1, age_band_2, age_band_3, age_band_4]
    ):
        # Plotting for each ageband
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))

        rects1 = ax1.bar(ind, age_band[0], width, color="#2ca02c")
        rects2 = ax1.bar(ind + width + xtra_space, age_band[1], width, color="#17becf")

        ax1.set_title(
            "Reactons from pregnant and non-pregant women of ages "
            + title
            + " years vaccinated with 1st dose against COVID-19",
            size=8,
        )

        rects1 = ax2.bar(ind, age_band[2], width, color="#2ca02c")
        rects2 = ax2.bar(ind + width + xtra_space, age_band[3], width, color="#17becf")

        ax2.set_title(
            "Reactons from pregnant and non-pregant women of ages "
            + title
            + " years vaccinated with 2nd dose against COVID-19",
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
                size=6,
            )

        plt.show()


if __name__ == "__main__":
    reactions_pregnant_moderna_pfizer()
    reactions_pregnant_vs_non_pregnant()
