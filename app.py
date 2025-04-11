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
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
from streamlit_option_menu import option_menu
from catboost import CatBoostRegressor
import requests
import io
import os
import urllib.request

# Initialize scaler (moved to top level)
scaler = StandardScaler()

#-----------Web page setting-------------------#
page_title = "💊Dr Olarele"
page_icon = "🎗🧬⌬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

# Title of the app
image = 'images/logo.png'
st.image(image, use_column_width=True)

selected = option_menu(
    menu_title=page_title + " " + page_icon,
    options=['Home', 'Select Target', 'About', 'Contact'],
    icons=["house-fill", "capsule", "capsule", "envelope-fill"],
    default_index=0,
    orientation="horizontal"
)

targets = ['Click to select a target', 'ER', 'Aromatase', 'CDK2', 'Braf', 'PI3K', 
           'VEGFR2', 'mTOR', 'PARP1', 'AKT', 'ATM', 'FGFR1', 'PR', 'HDAC1', 
           'HDAC2', 'HDAC8', 'CXCR4', 'HER2', 'AR', 'JAK2', 'GSK3B']

if selected == "Home":
    st.markdown("""
    <h3 style='color: darkblue;'>Welcome to Breast Cancer pIC<sub>50</sub> Prediction Web App</h3>
    We are thrilled to have you here. This app is designed to help researchers, clinicians, and scientists predict the <strong>pIC<sub>50</sub> values</strong> of compounds targeting <strong>20 different breast cancer targets</strong>. Whether exploring potential drug candidates or analyzing molecular interactions, this tool is here to simplify your work and accelerate your discoveries.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h4 style='color: blue;'>Key Features</h4>
    <strong>...</strong>
    """, unsafe_allow_html=True)

    image2 = 'images/workflow.png'
    st.image(image2, use_column_width=True)
    
    with st.sidebar:
        st.header("Overview and Usage")
        st.markdown("""
        <h4 style='color: blue;'>Brief Overview of the App</h4>
        The <strong>Breast Cancer pIC<sub>50</sub> Predictor</strong> is a powerful tool that leverages advanced <em>machine learning algorithms</em> to predict the <strong>pIC<sub>50</sub> values</strong> of compounds. The pIC<sub>50</sub> value is a critical metric in drug discovery, representing the potency of a compound in inhibiting a specific target.<br>
               
        <h4 style='color: blue;'>How to Use the App</h4>
        <strong>1. Select a Target:</strong> Choose one of the 20 breast cancer targets from the Home page.<br>
        <strong>2. Input Your Compound:</strong> Upload compounds' SMILES string file.<br>
        <strong>3. Get Predictions:</strong> Click <strong>Predict</strong> to receive the pIC<sub>50</sub> value for your compound.<br>
        <strong>4. Explore Results:</strong> View detailed predictions and download the results for further analysis.<br>
              
        <h4 style='color: blue;'>Why Use This App?</h4>
        <strong>Save Time:</strong> Quickly screen compounds and prioritize the most potent candidates.<br>
        <strong>Data-Driven Decisions:</strong> Make informed decisions based on accurate pIC<sub>50</sub> predictions.<br>
        <strong>Accelerate Research:</strong> Streamline your drug discovery workflow and focus on the most promising leads.<br>
            
        <h4 style='color: blue;'>Get Started</h4>
        Ready to explore? Click on the <strong>Select Target</strong> button to begin your journey toward discovering potent breast cancer inhibitors. If you have any questions or need assistance, please contact us.
        """, unsafe_allow_html=True)
        st.markdown("""[Example input file](https://raw.githubusercontent.com/afolabiowoloye/xyz/refs/heads/main/sample.csv)""")

if selected == "Select Target":
    st.subheader("Select preferred target")
    selected_target = st.selectbox(picker_icon, targets)

    if selected_target == "Click to select a target":
        st.markdown("""
        <h7 style='color: red;'><strong>Note: </strong>pIC<sub>50</sub> is the negative log of the IC<sub>50</sub> value, offering a logarithmic measure of compound potency.</h7>
        """, unsafe_allow_html=True)
        
    if selected_target == "ER":
        @st.cache_resource
        def load_model():
            MODEL_URL = "https://github.com/afolabiowoloye/catboost/raw/refs/heads/main/model/catboost_regression_model.cbm"
            MODEL_PATH = "model.cbm"
    
            if not os.path.exists(MODEL_PATH):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
            model = CatBoostRegressor()
            model.load_model(MODEL_PATH)
            return model

        model = load_model()

        # File uploader for SMILES data
        smiles_file = st.file_uploader("Upload your sample.csv", type=["csv", "txt"])
        st.markdown("""[Example input file](https://raw.githubusercontent.com/afolabiowoloye/xyz/refs/heads/main/sample.csv)""")
    
        if smiles_file is not None:
            try:
                sample = pd.read_csv(smiles_file)
                if 'SMILES' not in sample.columns:
                    st.error("Error: The uploaded file must contain a 'SMILES' column.")
                    st.stop()
                    
                st.write("Sample Data Preview:")
                st.dataframe(sample.head())

                # Getting RDKit descriptors
                def RDKit_descriptors(SMILES):
                    mols = [Chem.MolFromSmiles(i) for i in SMILES]
                    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
                    desc_names = calc.GetDescriptorNames()
                    Mol_descriptors = []
                    for mol in mols:
                        if mol is None:
                            continue
                        mol = Chem.AddHs(mol)
                        descriptors = calc.CalcDescriptors(mol)
                        Mol_descriptors.append(descriptors)
                    return Mol_descriptors, desc_names

                MoleculeDescriptors_list, desc_names = RDKit_descriptors(sample['SMILES'])
                if not MoleculeDescriptors_list:
                    st.error("Error: No valid molecules found in the SMILES data.")
                    st.stop()
                    
                df_ligands_descriptors = pd.DataFrame(MoleculeDescriptors_list, columns=desc_names)
                
                # Drop problematic columns if they exist
                for col in ["SPS", "AvgIpc"]:
                    if col in df_ligands_descriptors.columns:
                        df_ligands_descriptors = df_ligands_descriptors.drop(col, axis=1)

                # Predictions
                df_ligands_descriptors_scaled = scaler.fit_transform(df_ligands_descriptors)
                sample['predicted_pIC50'] = model.predict(df_ligands_descriptors_scaled)
                
                st.write("Predicted pIC50 Values:")
                st.dataframe(sample[['name', 'SMILES', 'predicted_pIC50']])
                
                # Download button
                csv = sample.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="predicted_pIC50_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

