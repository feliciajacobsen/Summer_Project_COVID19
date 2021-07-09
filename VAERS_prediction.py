import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


def VAERS_data():
    data_path = Path("./archive")
    VAERS_path = data_path / "VAERS"

    data = pd.read_csv(
        VAERS_path / "2021VAERSDATA.csv", encoding="ISO-8859-1", low_memory=False
    )
    data = data.drop(
        [
            "RECVDATE",
            "CAGE_YR",
            "CAGE_MO",
            "RPT_DATE",
            "SYMPTOM_TEXT",
            "DIED",
            "DATEDIED",
            "L_THREAT",
            "ER_VISIT",
            "HOSPITAL",
            "HOSPDAYS",
            "X_STAY",
            "DISABLE",
            "RECOVD",
            "ONSET_DATE",
            "NUMDAYS",
            "LAB_DATA",
            "V_ADMINBY",
            "V_FUNDBY",
            "OTHER_MEDS",
            "CUR_ILL",
            "HISTORY",
            "PRIOR_VAX",
            "SPLTTYPE",
            "FORM_VERS",
            "TODAYS_DATE",
            "BIRTH_DEFECT",
            "OFC_VISIT",
            "ER_ED_VISIT",
            "ALLERGIES",
        ],
        axis=1,
    )

    # defines new data frame with removed men
    data_females = data[data.SEX == "F"]

    # making data frame including only fertile women
    data_fertile_females = data_females[
        (data_females.AGE_YRS >= 15.0) & (data_females.AGE_YRS <= 50.0)
    ]

    symptoms = pd.read_csv(
        VAERS_path / "2021VAERSSYMPTOMS.csv", encoding="ISO-8859-1", low_memory=False
    )
    symptoms = symptoms.drop(
        [
            "SYMPTOMVERSION1",
            "SYMPTOMVERSION2",
            "SYMPTOMVERSION3",
            "SYMPTOMVERSION4",
            "SYMPTOMVERSION5",
        ],
        axis=1,
    )

    #symptoms = symptoms[symptoms.VAERS_ID == data_fertile_females["VAERS_ID"]]

    vax = pd.read_csv(
        VAERS_path / "2021VAERSVAX.csv", encoding="ISO-8859-1", low_memory=False
    )

    vax = vax.drop(["VAX_LOT", "VAX_ROUTE", "VAX_SITE", "VAX_NAME"], axis=1)
    vax = vax[vax.VAX_TYPE == "COVID19"]
    #vax = vax[vax.VAERS_ID == data_fertile_females["VAERS_ID"]]

    return data_fertile_females, symptoms, vax


if __name__ == "__main__":
    data, symptoms, vax = VAERS_data()
    #print(len(data)) # 12801
    print(symptoms.head())
    symptoms_tot = pd.concat([symptoms.SYMPTOM1, symptoms.SYMPTOM2, symptoms.SYMPTOM3, symptoms.SYMPTOM4, symptoms.SYMPTOM5])
    #print(symptoms_tot.nunique()) # number of unique symptoms=4235, length of all symptom entries is 240550
    sorted = symptoms_tot.sort_values(ascending=True)
    unique_sorted_symptoms_tot = sorted.drop_duplicates()
    unique_sorted_symptoms_tot.to_csv('list_of_unique_symptoms_covid19.csv')
