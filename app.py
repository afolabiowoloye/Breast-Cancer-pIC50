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
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
#from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu # for setting up menu bar


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
image = 'logo/logo.png'
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
        data_link_id = "1C-cFzESEfJcEdWGDLLdVawQbgZvbU_rQ"
        data_link = f'https://drive.google.com/uc?id={data_link_id}'
        data = pd.read_csv(data_link)
        st.write("Data Preview:")
        st.dataframe(data.head())
        #col = data.columns
        #st.write(col)

        # Data preprocessing
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)

        X = data.drop('pIC50', axis=1)
        y = data['pIC50']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

        # Scaling
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Model training
        model = ExtraTreesRegressor(n_estimators=1200, max_features='sqrt', min_samples_leaf=1,
                                    min_samples_split=5, max_depth=30, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        #st.write(f'R-squared (R2) Score: {r2:.2f}')
        #st.write(f'Mean Absolute Error: {mae:.2f}')
        #st.write(f'Mean Squared Error: {mse:.2f}')
        #st.write(f'Root Mean Squared Error: {rmse:.2f}')


        # Plotting
        fig, ax = plt.subplots()
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3, 'color': 'blue'}, line_kws={'color':'red'}, ax=ax)
        ax.set_xlabel('Experimental pIC50')
        ax.set_ylabel('Predicted pIC50')
        ax.set_xlim(2, 11.5)
        ax.set_ylim(2, 11.5)
        plt.title('Actual vs Predicted pIC50')
        st.pyplot(fig)

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

# -

## Braf
    if selected_target == "Braf":
    # Link to the dataset on Google Drive
        data_link_id = "1k57-I0CsiMpXko4mZV6GRpxoag9USJbf"
        data_link = f'https://drive.google.com/uc?id={data_link_id}'
        data = pd.read_csv(data_link)
        st.write("Data Preview:")
        st.dataframe(data.head())
        #col = data.columns
        #st.write(col)

        # Data preprocessing
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)

        X = data.drop('pIC50', axis=1)
        y = data['pIC50']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

        # Scaling
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Model training
        model = ExtraTreesRegressor(max_depth = 40,
                                min_samples_leaf = 1,
                                min_samples_split = 10,
                                n_estimators = 200,
                                random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        #st.write(f'R-squared (R2) Score: {r2:.2f}')
        #st.write(f'Mean Absolute Error: {mae:.2f}')
        #st.write(f'Mean Squared Error: {mse:.2f}')
        #st.write(f'Root Mean Squared Error: {rmse:.2f}')


        # Plotting
        fig, ax = plt.subplots()
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3, 'color': 'indigo'}, line_kws={'color':'crimson'}, ax=ax)
        ax.set_xlabel('Experimental pIC50')
        ax.set_ylabel('Predicted pIC50')
        ax.set_xlim(3.5, 10.5)
        ax.set_ylim(3.5, 10.5)
        plt.title('Actual vs Predicted pIC50')
        st.pyplot(fig)

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



    ## CDK2
    if selected_target == "CDK2":
        # Link to the dataset on Google Drive
        data_link_id = "1_m9ngbxP-jQnkspEinnj2Pd3GfC3S5ve"
        data_link = f'https://drive.google.com/uc?id={data_link_id}'
        data = pd.read_csv(data_link)
        st.write("Data Preview:")
        st.dataframe(data.head())
        #col = data.columns
        #st.write(col)

        # Data preprocessing
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)

        X = data.drop('pIC50', axis=1)
        y = data['pIC50']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

        # Scaling
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Model training
        model = LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        #st.write(f'R-squared (R2) Score: {r2:.2f}')
        #st.write(f'Mean Absolute Error: {mae:.2f}')
        #st.write(f'Mean Squared Error: {mse:.2f}')
        #st.write(f'Root Mean Squared Error: {rmse:.2f}')


        # Plotting
        fig, ax = plt.subplots()
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3, 'color': 'red'}, line_kws={'color':'blue'}, ax=ax)
        ax.set_xlabel('Experimental pIC50')
        ax.set_ylabel('Predicted pIC50')
        ax.set_xlim(2.5, 10)
        ax.set_ylim(3, 10)
        plt.title('Actual vs Predicted pIC50')
        st.pyplot(fig)

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


    ## Aromatase
    if selected_target == "Aromatase":
    # Link to the dataset on Google Drive
        data_link_id = "1kqoIyXMI4uBi4jHTw-gnjaq7AiWCFYZ0"
        data_link = f'https://drive.google.com/uc?id={data_link_id}'
        data = pd.read_csv(data_link)
        st.write("Data Preview:")
        st.dataframe(data.head())
        #col = data.columns
        #st.write(col)

        # Data preprocessing
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)

        X = data.drop('pIC50', axis=1)
        y = data['pIC50']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

        # Scaling
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Model training
        model = ExtraTreesRegressor(max_depth = 40,
                                min_samples_leaf = 1,
                                min_samples_split = 10,
                                n_estimators = 200,
                                random_state=42)
        model.fit(X_train_scaled, y_train)


        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        #st.write(f'R-squared (R2) Score: {r2:.2f}')
        #st.write(f'Mean Absolute Error: {mae:.2f}')
        #st.write(f'Mean Squared Error: {mse:.2f}')
        #st.write(f'Root Mean Squared Error: {rmse:.2f}')


        # Plotting
        fig, ax = plt.subplots()
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3, 'color': 'darkgreen'}, line_kws={'color':'deeppink'}, ax=ax)
        ax.set_xlabel('Experimental pIC50')
        ax.set_ylabel('Predicted pIC50')
        ax.set_xlim(3.5, 10.5)
        ax.set_ylim(3.5, 10.5)
        plt.title('Actual vs Predicted pIC50')
        st.pyplot(fig)

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


