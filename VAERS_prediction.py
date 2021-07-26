import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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
        data_path_2020 / "2020VAERSSYMPTOMS.csv",
        encoding="ISO-8859-1",
        low_memory=False,
    )
    symptoms_2021 = pd.read_csv(
        data_path_2021 / "2021VAERSSYMPTOMS.csv",
        encoding="ISO-8859-1",
        low_memory=False,
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
    patients.OTHER_MEDS[patients.OTHER_MEDS == "None"] = 0.0
    patients.OTHER_MEDS[patients.OTHER_MEDS == "Unknown"] = 0.0
    patients.OTHER_MEDS[patients.OTHER_MEDS == "Na"] = 0.0
    patients.OTHER_MEDS[patients.OTHER_MEDS == "NaN"] = 0.0
    patients.OTHER_MEDS[patients.OTHER_MEDS.notnull()] = 1.0  # not nan
    patients.OTHER_MEDS[patients.OTHER_MEDS.isnull()] = 0.0  # nan

    patients.CUR_ILL[patients.CUR_ILL == "None"] = 0.0
    patients.CUR_ILL[patients.CUR_ILL == "Unknown"] = 0.0
    patients.CUR_ILL[patients.CUR_ILL == "Na"] = 0.0
    patients.OTHER_MEDS[patients.CUR_ILL == "NaN"] = 0.0
    patients.CUR_ILL[patients.CUR_ILL.notnull()] = 1.0  # not nan
    patients.CUR_ILL[patients.CUR_ILL.isnull()] = 0.0  # nan

    patients.HOSPITAL[patients.HOSPITAL == "Y"] = 1.0
    patients.HOSPITAL[patients.HOSPITAL != 1] = 0.0

    patients.L_THREAT[patients.L_THREAT == "Y"] = 1.0
    patients.L_THREAT[patients.L_THREAT != 1] = 0.0

    patients.BIRTH_DEFECT[patients.BIRTH_DEFECT == "Y"] = 1.0
    patients.BIRTH_DEFECT[patients.BIRTH_DEFECT != 1] = 0.0

    patients.DISABLE[patients.DISABLE == "Y"] = 1.0
    patients.DISABLE[patients.DISABLE != 1] = 0.0

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
    fertile_females = females[(females.AGE_YRS >= 15.0) & (females.AGE_YRS <= 50.0)]

    # drop invalid entries to number of doses
    fertile_females = fertile_females[fertile_females.VAX_DOSE_SERIES != "UNK"]
    fertile_females = fertile_females[fertile_females.VAX_DOSE_SERIES != "NaN"]
    fertile_females = fertile_females[fertile_females.VAX_DOSE_SERIES != "7+"]
    fertile_females = fertile_females[fertile_females.VAX_DOSE_SERIES != ""]
    fertile_females.dropna(subset=["VAX_DOSE_SERIES"], inplace=True)
    # turn number of doses from object type to float
    pd.to_numeric(fertile_females.VAX_DOSE_SERIES)

    fertile_females.reset_index(drop=True, inplace=True)

    # one-hot encode vaccine manufacturer
    one_hot = pd.get_dummies(fertile_females.VAX_MANU)
    # join the encoded df
    fertile_females = fertile_females.join(one_hot)
    # drop column VAX_MANU as it is now encoded
    fertile_females.drop(["VAX_MANU"], axis=1, inplace=True)

    # list of heart- and hormone-related symptoms (defined in separate .txt-file)
    with open("symptoms.txt", "r") as infile:
        symptom_columns = infile.readlines()
        infile.close()

    # removing "\n"-character from list
    symptom_columns = [x.strip() for x in symptom_columns]

    # making empty dataframe of symptoms
    symptoms_df = pd.DataFrame(
        data=np.zeros((len(fertile_females), len(symptom_columns))),
        columns=symptom_columns,
    )

    # concatenating patient info and symptoms of interest
    fertile_females = pd.concat([fertile_females, symptoms_df], axis=1)

    # one-hot encode the symptoms of interest and remove the rest
    for i, id in enumerate(fertile_females.VAERS_ID):
        reports = symptoms[
            symptoms.VAERS_ID == id
        ]  # symptom reports must match with correct patient
        # loop over each report for each patient
        for index, report in reports.iterrows():
            # loop over each symptom in a report
            for symptom_number in [
                "SYMPTOM1",
                "SYMPTOM2",
                "SYMPTOM3",
                "SYMPTOM4",
                "SYMPTOM5",
            ]:
                symptom = report[symptom_number]
                # check if symptom is equal to the list of symptom of interest
                if symptom in symptom_columns:
                    fertile_females.loc[fertile_females.VAERS_ID == id, symptom] = 1.0

    # find number of symptom of interest
    N = len(symptom_columns)

    # drop patient ID
    fertile_females.drop(["VAERS_ID"], axis=1, inplace=True)

    # return separate patient info dataframe and a dataframe for symptom labels
    return fertile_females.iloc[:, :-N], fertile_females.iloc[:, -N:]


def save_preprocessed_data():
    X, y = read_and_preprocess_VAERS_data()
    data_path = Path("./data/preprocessed")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    X.to_csv(data_path / "patient_data.csv", index=False)
    y.to_csv(data_path / "symptoms_data.csv", index=False)

    return None


def load_preprocessed_data():
    X = pd.read_csv("./data/preprocessed/patient_data.csv")
    y = pd.read_csv("./data/preprocessed/symptoms_data.csv")

    return X, y


def run_ml_model(X, y):
    # print(X.head(20))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = RandomForestClassifier(
        bootstrap=True, max_depth=10, max_features="sqrt", random_state=1
    )
    rfc.fit(X_train, y_train)
    print(accuracy_score(y_test, rfc.predict(X_test))) #0.9197739678634581 WHAT!!


if __name__ == "__main__":
    # X, y = read_and_preprocess_VAERS_data()

    # When I need to save preprocess new data
    # save_preprocessed_data() # 5 sek

    # When I can re-use the already preprocessed data
    X, y = load_preprocessed_data()

    # When I need to run prediction model
    run_ml_model(X, y)
