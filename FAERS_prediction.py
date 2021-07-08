import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


def VAERS_data():
    data_path = Path("./data")
    FAERS_path = data_path / "FAERS"

    data = pd.read_xml(
        FAERS_path / "1_ADR21Q1_format.xml"
    )

    #data = data[data.reactionmeddrapt == "COVID19"]

    return data

data = VAERS_data()
for col in data.columns:
    print(col)
