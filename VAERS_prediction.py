import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


def VAERS_data():
    data_path_2021 = Path("./data/2021VAERSData")


    patients = pd.read_csv(
        data_path_2021 / "2021VAERSDATA.csv", encoding="ISO-8859-1", low_memory=False
    )
    patients = patients.drop(
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
    #print(len(patients)) #397444

    symptoms = pd.read_csv(
        data_path_2021 / "2021VAERSSYMPTOMS.csv", encoding="ISO-8859-1", low_memory=False
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

    vax = pd.read_csv(
        data_path_2021 / "2021VAERSVAX.csv", encoding="ISO-8859-1", low_memory=False
    )

    vax = vax.drop(["VAX_LOT", "VAX_ROUTE", "VAX_SITE", "VAX_NAME"], axis=1)

    patients_symptoms = pd.merge(patients, symptoms, on="VAERS_ID")
    data = pd.merge(patients_symptoms, vax, on="VAERS_ID")

    # defines new data frame with removed men
    data_females = data[data.SEX == "F"]
    data_females = data_females[data_females.VAX_TYPE == "COVID19"]
    data_females = data_females.drop(["VAX_TYPE", "SEX"], axis=1)

    # making data frame including only fertile women
    data_fertile_females = data_females[
        (data_females.AGE_YRS >= 15.0) & (data_females.AGE_YRS <= 50.0)
    ]


    return data_fertile_females, symptoms, vax


if __name__ == "__main__":
    data, symptoms, vax = VAERS_data()
    print(len(data)) # 199157
    #print(data.head())
    symptoms_tot = pd.concat([symptoms.SYMPTOM1, symptoms.SYMPTOM2, symptoms.SYMPTOM3, symptoms.SYMPTOM4, symptoms.SYMPTOM5])
    #print(symptoms_tot.nunique()) # number of unique symptoms=4235, length of all symptom entries is 240550
    sorted = symptoms_tot.sort_values(ascending=True)
    unique_sorted_symptoms_tot = sorted.drop_duplicates()
    #unique_sorted_symptoms_tot.to_csv('list_of_unique_symptoms_covid19.csv')
