import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    make_scorer,
)

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

    # list of heart related symptoms (defined in separate .txt-file)
    with open("heart_symptoms.txt", "r") as infile:
        heart_symptom_columns = infile.readlines()
        infile.close()

    # list of  hormone-related symptoms (defined in separate .txt-file)
    with open("hormone_symptoms.txt", "r") as infile:
        hormone_symptom_columns = infile.readlines()
        infile.close()

    # removing "\n"-character from list
    heart_symptom_columns, hormone_symptom_columns = [x.strip() for x in heart_symptom_columns], [x.strip() for x in hormone_symptom_columns]

    # empty data frame with two labels (hormone and heart symptoms)
    symptoms_df = pd.DataFrame(
        data=np.zeros((len(fertile_females), 2)),
        columns=["Heart related symptoms", "Hormone related symptoms"],
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
                if symptom in heart_symptom_columns:
                    fertile_females.loc[fertile_females.VAERS_ID==id, "Heart related symptoms"] = 1.0

                if symptom in hormone_symptom_columns:
                    fertile_females.loc[fertile_females.VAERS_ID==id, "Hormone related symptoms"] = 1.0

    # drop patient ID
    fertile_females.drop(["VAERS_ID"], axis=1, inplace=True)

    # return separate patient info dataframe and a dataframe for symptom labels
    return fertile_females.iloc[:, :-2], fertile_females.iloc[:, -2:]


def save_preprocessed_data():
    """
    Function runs read_and_preprocess_VAERS_data() saves the preprocessed data.
    Patient data (input data) and symptom data (target data) are stored into two
    separate .csv-files.

    Params:
    -------
        None

    Returns:
    -------
        None
    """
    X, y = read_and_preprocess_VAERS_data()
    data_path = Path("./data/preprocessed")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    X.to_csv(data_path / "patient_data.csv", index=False)
    y.to_csv(data_path / "symptoms_data.csv", index=False)

    return None


def load_preprocessed_data():
    """
    Function runs read_and_preprocess_VAERS_data() saves the preprocessed data.
    Patient data (input data) and symptom data (target data) are stored into two
    separate .csv-files.

    Params:
    -------
        None

    Returns:
    -------
        X: Pandas dataframe
            Dataframe of patient info. Each row represent one patient.
        y: Pandas dataframe
            One-hot encoded 1d array for each patient. The vector represents
            a combination of heart- and hormone related symptoms.
    """

    X = pd.read_csv("./data/preprocessed/patient_data.csv")
    y = pd.read_csv("./data/preprocessed/symptoms_data.csv")

    return X, y

def reverse_one_hot_encoding(y):
    """
    Function reverse one hot encoded multi-label targets. There are a total of
    four unique combinations of the two labeled targets

    class 1 = no heart- og hormone-related symptoms [0.0,0.0]
    class 2 = heart symptom [1.0,0.0]
    class 3 = hormone symptom [0.0,1.0]
    class 4 = both symptoms [1.0,1.0]

    Params:
    -------
        y: pandas DataFrame
            Must contain 2 columns of one-hot encoded multilabel targets

    Returns:
    -------
        y: Numpy array
            1D numpy arrray containing a total of four unique classes

    """

    y_new = np.zeros(y.shape[0])

    if type(y) != pd.DataFrame:
        y = pd.DataFrame(
                data=y,
                columns=["Heart related symptoms", "Hormone related symptoms"],
                )


    for i, tup in enumerate(y.iterrows()):
        idx, r = tup
        if (r.iloc[0]==0.0 and r.iloc[1]==0.0):
            y_new[i]=1.0
        elif (r.iloc[0]==1.0 and r.iloc[1]==0.0):
            y_new[i] = 2.0
        elif (r.iloc[0]==0.0 and r.iloc[1]==1.0):
            y_new[i] = 3.0
        elif (r.iloc[0]==1.0 and r.iloc[1]==1.0):
            y_new[i] = 4.0

    return y_new


def one_hot_encoding(y):
    """
    Function one hot encodes multi-label targets. There are a total of
    four unique combinations of the two labeled targets

    class 1 = no heart- og hormone-related symptoms [0.0,0.0]
    class 2 = heart symptom [1.0,0.0]
    class 3 = hormone symptom [0.0,1.0]
    class 4 = both symptoms [1.0,1.0]

    Params:
    -------
        y: Numpy array
            1D numpy array

    Returns:
    -------
        y: Numpy array
            Contains 2 columns of one-hot encoded multilabel targets

    """

    y_new = np.zeros((y.shape[0], 2))

    for i, c in enumerate(y):
        if c==1.0:
            y_new[i,0]= 0.0
            y_new[i,0]= 0.0
        elif c==2.0:
            y_new[i,0]= 1.0
            y_new[i,1]= 0.0
        elif c==3.0:
            y_new[i,0]= 0.0
            y_new[i,1]= 1.0
        elif c==4.0:
            y_new[i,0]= 1.0
            y_new[i,1]= 1.0

    return y_new


def downsample_data(X, y):
    """
    Function randomly downsample 60% of the original data.

    Params:
    -------
        X: pandas DataFrame
            Input data
        y: pandas DataFrame
            2D pandas dataframe containing target labels

    Returns:
    -------
        X: pandas DataFrame
            downsampled input data
        y: pandas DataFrame
            downsampled array of target labels
    """

    # randomly downsample rows with under-represented target labels
    no_symptoms = y.loc[(y["Heart related symptoms"]==0.0) & (y["Hormone related symptoms"]==0.0)]
    random_idx = no_symptoms.sample(frac=0.6).sort_index().index.tolist()

    return X.drop(index=random_idx), y.drop(index=random_idx)


def upsample_data(X, y):
    """
    Function randomly upsample input and target data such that all classes are
    equally represented as majority class.

    Params:
    -------
        X: pandas DataFrame
            Input data
        y: pandas DataFrame
            2D pandas dataframe containing target labels

    Returns:
    -------
        X: pandas DataFrame
            downsampled input data
        y: pandas DataFrame
            downsampled array of target labels
    """
    # reverse one-hot encoding in order to use RandomOverSampler
    y = reverse_one_hot_encoding(y)

     # upsample all classes except for the majority class
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=1)
    X_res, y_res = ros.fit_resample(X, y)

    # return to one-hot encoded multilabel targets
    y_res = one_hot_encoding(y_res)

    # make target array into pandas dataframe
    y_res = pd.DataFrame(
        data=y_res,
        columns=["Heart related symptoms", "Hormone related symptoms"],
        )

    return X_res, y_res


def run_ml_model(X, y):
    """
    Function performs multi-label classification with the Random Forest
    classifier from the Scikit-learn library.

    Function prints the recall score and saves confusion matrix and a histogram of
    relative feature importances in a separate folder named "figures".


    Params:
    -------
        X: pandas DataFrame
            Input data
        y: pandas DataFrame
            2D pandas dataframe containing target labels

    Returns:
    -------
        None
    """
    # load dataset and split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # upsample all other classes except majority class
    X_train, y_train = upsample_data(X_train, y_train)

    # define model
    rfc = RandomForestClassifier(n_estimators=250,bootstrap=True, random_state=1, max_depth=2, max_features="sqrt")


    # train on training data
    rfc.fit(X_train, y_train)

    # predict symptoms on test input data
    y_pred = rfc.predict(X_test)

    # reverse one-hot encoding in order to get recall and confusion matrix
    y_test_new, y_pred_new = reverse_one_hot_encoding(y_test), reverse_one_hot_encoding(y_pred)

    # print performance scores based model prediction on test input and true test output
    recall = recall_score(y_test, y_pred, average="weighted")
    print(f"Recall = {recall:2.4f}")

    if not os.path.exists(Path("./figures")):
         os.makedirs(Path("./figures"))

    # Plot confusion matrix
    plt.figure(figsize=(7,7))
    plt.title("Accuracy scores of vaccinated femlaes (COVID-19)\n of maternal age from VAERS dataset")
    labels = ["No SOI","Heart related","Hormone related","Both"]
    sns.heatmap(
        confusion_matrix(y_test_new, y_pred_new),
        cmap="Blues",
        annot=True,
        fmt="d",
        yticklabels=labels,
        xticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig("./figures/confusion_matrix.png")

    # Plot histogram of feature importancess
    plt.figure(figsize=(12,5))
    features = X_train.columns # list of feature names
    importances = rfc.feature_importances_ # get feature importances
    indices = np.argsort(importances)
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="#1f77b4", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=6)
    plt.xlabel("Relative Importance")
    plt.savefig("./figures/feature_importances.png")



if __name__ == "__main__":

    # When I need to save preprocess new data
    if not os.path.exists(Path("./data/preprocessed")):
        save_preprocessed_data()

    # When I can re-use the already preprocessed data
    X, y = load_preprocessed_data()

    # When I need to run prediction model
    run_ml_model(X, y)
