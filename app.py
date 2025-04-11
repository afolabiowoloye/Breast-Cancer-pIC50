# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
from streamlit_option_menu import option_menu # for setting up menu bar
from catboost import CatBoostRegressor
np.__version__ = '1.21.6'
import requests
import io


# +
#-----------Web page setting-------------------#
page_title = "💊Breast Cancer pIC50 Prediction Web App"
page_icon = "🎗🧬⌬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
layout = "centered"
#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

# Title of the app
#st.title("pIC50 Prediction App")
# Logo image
image = 'images/logo.png'
st.image(image, use_container_width=True)


# -

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Select Target', 'About', 'Contact'],
    icons = ["house-fill", "capsule",  "capsule", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

targets = ['Click to select a target', 'ER', 'Aromatase', 'CDK2', 'Braf', 'PI3K', 'VEGFR2', 'mTOR', 'PARP1', 'AKT', 'ATM', 'FGFR1', 'PR', 'HDAC1', 'HDAC2', 'HDAC8', 'CXCR4', 'HER2', 'AR', 'JAK2', 'GSK3B']


# +
if selected == "Home":
    st.markdown("""
    <h3 style='color: darkblue;'>Welcome to Breast Cancer pIC<sub>50</sub> Prediction Web App</h3>
    We are thrilled to have you here. This app is designed to help researchers, clinicians, and scientists predict the <strong>pIC<sub>50</sub> values</strong> of compounds targeting <strong>20 different breast cancer targets</strong>. Whether exploring potential drug candidates or analyzing molecular interactions, this tool is here to simplify your work and accelerate your discoveries.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h4 style='color: blue;'>Key Features</h4>
    <strong>...</strong>
    """, unsafe_allow_html=True)

    #<strong>20 Breast Cancer Targets:</strong> Predict pIC<sub>50</sub> values for compounds targeting a wide range of breast cancer-related proteins, including kinases, receptors, and enzymes.<br>
    #<strong>User-Friendly Interface:</strong> Simply input your compound's details (e.g., SMILES string or molecular structure), and the app will generate predictions instantly.<br>
    #<strong>Reliable Predictions:</strong> Built on robust machine learning models trained on high-quality datasets, the app delivers reliable and actionable insights.<br>
    #<strong>Research-Ready:</strong> Designed to support drug discovery and molecular research, helping you identify promising compounds and optimize drug candidates.<br>
    
    image2 = 'logo/workflow.png'
    st.image(image2, use_container_width=True)
    
    with st.sidebar.header("""Overview and Usage"""):
        st.sidebar.markdown("""
        <h4 style='color: blue;'>Brief Overview of the App</h4>
        The <strong>Breast Cancer pIC<sub>50</sub> Predictor</strong> is a powerful tool that leverages advanced <em>machine learning algorithms</em> to predict the <strong>pIC<sub>50</sub> values</strong> of compounds. The pIC<sub>50</sub> value is a critical metric in drug discovery, representing the potency of a compound in inhibiting a specific target.<br>
               
        <h4 style='color: blue;'>How to Use the App</h4>
        <strong>1. Select a Target:</strong> Choose one of the 20 breast cancer targets from the Home page.<br>
        <strong>2. Input Your Compound:</strong> Upload compounds' SMILES string file.<br>
        <strong>3. Get Predictions:</strong> Click <strong>Predict</strong> to receive the <sub>50</sub> value for your compound.<br>
        <strong>4. Explore Results:</strong> View detailed predictions and download the results for further analysis.<br>
              
        <h4 style='color: blue;'>Why Use This App?</h4>
        <strong>Save Time:</strong> Quickly screen compounds and prioritize the most potent candidates.<br>
        <strong>Data-Driven Decisions:</strong> Make informed decisions based on accurate pIC<sub>50</sub> predictions.<br>
        <strong>Accelerate Research:</strong> Streamline your drug discovery workflow and focus on the most promising leads.<br>
            
        <h4 style='color: blue;'>Get Started</h4>
        Ready to explore? Click on the <strong>Select Target</strong> button to begin your journey toward discovering potent breast cancer inhibitors. If you have any questions or need assistance, please contact us.
        """, unsafe_allow_html=True)
        st.markdown("""[Example input file](https://raw.githubusercontent.com/afolabiowoloye/xyz/refs/heads/main/sample.csv)""")


     
# Display data preview
#st.write("Data Preview:")
#st.dataframe(df.head())

if selected == "Select Target":
    st.subheader("Select preferred target")
    selected_target = st.selectbox(picker_icon, targets)

    # ER Dataset Training
    if selected_target == "Click to select a target":
        st.markdown("""
        <h7 style='color: red;'><strong>Note: </strong>pIC<sub>50</sub> is the negative log of the IC<sub>50</sub> value, offering a logarithmic measure of compound potency.</h7>
        """, unsafe_allow_html=True)
        
    if selected_target == "ER":
        def load_model():
            import os
            import urllib.request
            MODEL_URL = "https://github.com/afolabiowoloye/catboost/raw/refs/heads/main/model/catboost_regression_model.cbm"
            MODEL_PATH = "model.cbm"
    
            if not os.path.exists(MODEL_PATH):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
            model = CatBoostRegressor()
            return model.load_model(MODEL_PATH)

        model = load_model()

        # File uploader for SMILES data
        smiles_file = st.file_uploader("Upload your sample.csv", type="csv")
        st.markdown("""[Example input file](https://raw.githubusercontent.com/afolabiowoloye/xyz/refs/heads/main/sample.csv)""")
    
        if smiles_file is not None:
            sample = pd.read_csv(smiles_file)
            st.write("Sample Data Preview:")
            st.dataframe(sample.head())


        # Getting RDKit descriptors
            def RDKit_descriptors(SMILES):
                mols = [Chem.MolFromSmiles(i) for i in SMILES]
                calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
                desc_names = calc.GetDescriptorNames()
                Mol_descriptors = []
                for mol in mols:
                    mol = Chem.AddHs(mol)
                    descriptors = calc.CalcDescriptors(mol)
                    Mol_descriptors.append(descriptors)
                return Mol_descriptors, desc_names

            MoleculeDescriptors_list, desc_names = RDKit_descriptors(sample['SMILES'])
            df_ligands_descriptors = pd.DataFrame(MoleculeDescriptors_list, columns=desc_names)
            #st.dataframe(df_ligands_descriptors.head())
        
            df_ligands_descriptors = df_ligands_descriptors.drop(["SPS", "AvgIpc"], axis=1)
            #col2 = df_ligands_descriptors.columns
            #st.write(col2)

        # Predictions
            df_ligands_descriptors_scaled = scaler.fit_transform(df_ligands_descriptors)
            sample['predicted_pIC50'] = model.predict(df_ligands_descriptors_scaled)
            st.write("Predicted pIC50 Values:")
            st.dataframe(sample[['name', 'SMILES', 'predicted_pIC50']])
            download_result = pd.DataFrame(sample)
            download_result = download_result.to_csv(index=False)
            st.download_button("Press to Download Result",download_result,"file.csv","text/csv",key='download-csv')

