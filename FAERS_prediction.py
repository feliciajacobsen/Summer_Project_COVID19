import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def VAERS_data():
    data_path = Path("./archive/faers_xml_2021Q1")
    FAERS_path = data_path / "XML"
    
    tree = ET.parse(
        FAERS_path / "1_ADR21Q1_format.xml"
    )
    root = tree.getroot()
    """
    for x in root.findall("safetyreport"):
        print(x.tag, x.attrib)
    """
    for x in root[1].findall("patient"):
        age = x.find("patientonsetage").text
        sex = x.find("patientsex").text
        print(age, sex)
    
    
   

VAERS_data()