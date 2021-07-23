import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
pd.options.mode.chained_assignment = None


def read_and_preprocess_VAERS_data():
    """
    Function reads VAERS data from 2020 and later, and drops columns
    which is not suitable for prediction. All non-Covid-19 are omitted.
    Data is one-hot encoded where only 115 symptoms of interest are kept
    in the dataset.

    The symptoms of interest are:
        - heart-related and hormone-related symptoms.

    Params:
    -------
        None

    Returns:
    -------
        X: pandas DataFrame
            Covid-19 vaccinated patient data.
        y: pandas DataFrame
            one-hot encoded labels of symptoms of interest.
    """
    # define data paths for each year
    data_path_2021 = Path("./data/2021VAERSData")
    data_path_2020 = Path("./data/2020VAERSData")

    # read data files
    patients_2021 = pd.read_csv(
    data_path_2021 / "2021VAERSDATA.csv", encoding="ISO-8859-1", low_memory=False
    )
    patients_2020 = pd.read_csv(

    data_path_2020 / "2020VAERSData.csv", encoding="ISO-8859-1", low_memory=False
    )
    symptoms_2020 = pd.read_csv(
        data_path_2020 / "2020VAERSSYMPTOMS.csv", encoding="ISO-8859-1", low_memory=False
    )
    symptoms_2021 = pd.read_csv(
        data_path_2021 / "2021VAERSSYMPTOMS.csv", encoding="ISO-8859-1", low_memory=False
    )

    vax_2020 = pd.read_csv(
        data_path_2020 / "2020VAERSVAX.csv", encoding="ISO-8859-1", low_memory=False
    )
    vax_2021 = pd.read_csv(
        data_path_2021 / "2021VAERSVAX.csv", encoding="ISO-8859-1", low_memory=False
    )

    # concatenate data from each year
    patients = pd.concat([patients_2020, patients_2021], axis=0)
    symptoms = pd.concat([symptoms_2020, symptoms_2021], axis=0)
    vax = pd.concat([vax_2020, vax_2021], axis=0)

    # drop columns from each dataframe which is not of interest
    patients = patients.drop(
        [
            "RECVDATE",
            "STATE",
            "VAX_DATE",
            "CAGE_YR",
            "CAGE_MO",
            "RPT_DATE",
            "SYMPTOM_TEXT",
            "DIED",
            "HISTORY",
            "DATEDIED",
            "ER_VISIT",
            "HOSPDAYS",
            "X_STAY",
            "RECOVD",
            "ONSET_DATE",
            "NUMDAYS",
            "LAB_DATA",
            "V_ADMINBY",
            "V_FUNDBY",
            "PRIOR_VAX",
            "SPLTTYPE",
            "FORM_VERS",
            "TODAYS_DATE",
            "OFC_VISIT",
            "ER_ED_VISIT",
            "ALLERGIES",
        ],
        axis=1,
    )

    # One-hot-encoding: yes = 1, no = 0
    patients.OTHER_MEDS[patients.OTHER_MEDS=="None"] = 0
    patients.OTHER_MEDS[patients.OTHER_MEDS=="Unknown"] = 0
    patients.OTHER_MEDS[patients.OTHER_MEDS=="Na"] = 0
    patients.OTHER_MEDS[patients.OTHER_MEDS.notnull()] = 1  # not nan
    patients.OTHER_MEDS[patients.OTHER_MEDS.isnull()] = 0   # nan

    patients.CUR_ILL[patients.CUR_ILL=="None"] = 0
    patients.CUR_ILL[patients.CUR_ILL=="Unknown"] = 0
    patients.CUR_ILL[patients.CUR_ILL=="Na"] = 0
    patients.CUR_ILL[patients.CUR_ILL.notnull()] = 1  # not nan
    patients.CUR_ILL[patients.CUR_ILL.isnull()] = 0   # nan

    patients.HOSPITAL[patients.HOSPITAL=="Y"] = 1
    patients.HOSPITAL[patients.HOSPITAL!=1] = 0

    patients.L_THREAT[patients.L_THREAT=="Y"] = 1
    patients.L_THREAT[patients.L_THREAT!=1] = 0

    patients.BIRTH_DEFECT[patients.BIRTH_DEFECT=="Y"] = 1
    patients.BIRTH_DEFECT[patients.BIRTH_DEFECT!=1] = 0

    patients.DISABLE[patients.DISABLE=="Y"] = 1
    patients.DISABLE[patients.DISABLE!=1] = 0

    # drop columns which is not of interest
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

    # drop columns which is not of interest
    vax = vax.drop(["VAX_LOT", "VAX_ROUTE", "VAX_SITE", "VAX_NAME"], axis=1)

    # merge patients and vaccine data depending on ID number
    patients_vax = pd.merge(patients, vax, on="VAERS_ID")

    # defines new data frame with removed men
    females = patients_vax[patients_vax.SEX == "F"]

    # only include data from covid-19 vaccine
    females = females[females.VAX_TYPE == "COVID19"]

    # remove columns of sex (only females), vaccine type (only covid19) and state (uninteresting for now)
    females = females.drop(["VAX_TYPE", "SEX"], axis=1)

    # making data frame including only fertile women
    fertile_females = females[
        (females.AGE_YRS >= 15.0) & (females.AGE_YRS <= 50.0)
    ]

    fertile_females.reset_index(drop=True, inplace=True)

    # list of heart- and hormone-related symptoms (defined in separate .txt-file)
    with open("symptoms.txt", "r") as infile:
        symptom_columns = infile.readlines()
        infile.close()

    # removing "\n"-character from list
    symptom_columns = [x.strip() for x in symptom_columns]

    #print(len(symptom_columns)) # 115 symptoms of interest in total

    # making empty dataframe of symptoms
    symptoms_df = pd.DataFrame(data=np.zeros((len(fertile_females), len(symptom_columns))), columns=symptom_columns)

    # concatenating patient info and symptoms of interest
    fertile_females = pd.concat([fertile_females, symptoms_df], axis=1)

    # one-hot encode the symptoms of interest and remove the rest
    """
    for i, id in enumerate(fertile_females.VAERS_ID):
        # all reports of fertile females vaccinated from covid-19
        reports = symptoms[symptoms.VAERS_ID == id]
        for index, report in reports.iterrows():
            for symptom_number in ["SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"]:
                symptom = report[symptom_number]
                if symptom in fertile_females:
                    fertile_females[fertile_females.VAERS_ID == id][symptom] = 1.0
    """

    for i, id in enumerate(fertile_females.VAERS_ID):
        reports = symptoms[symptoms.VAERS_ID == id] # symptom reports must match with correct patient
        # loop over each report for each patient
        for index, report in reports.iterrows():
            # loop over each symptom in a row
            for symptom_number in ["SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"]:
                symptom = report[symptom_number]
                if symptom in symptom_columns:
                    #print("found symptom of interest")
                    fertile_females[fertile_females.VAERS_ID == id][symptom] = 1.0

    N = len(symptom_columns)
    return fertile_females.iloc[:,:-N], fertile_females.iloc[:,-N:]



def save_preprocessed_data():
    X, y = read_and_preprocess_VAERS_data()
    data_path = Path("./data/preprocessed")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    X.to_csv(data_path / "patient_data.csv")
    y.to_csv(data_path / "symptoms_data.csv")



def prediction_boost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    abc = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R")
    abc.fit(X_train, y_train)
    print(accuracy_score(y_test, abc.predict(X_test)))



if __name__ == "__main__":
    # When I need to preprocess new data
    #X, y = read_and_preprocess_VAERS_data()
    save_preprocessed_data() # 5 sek

    # When I can re-use the already preprocessed data
    #load_preprocessed_data() # 5 sek

    #run_ml_model()
    #print(len(data)) # 199157 for 2021, 7937 for 2020
