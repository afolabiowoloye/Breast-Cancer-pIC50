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
import os
import urllib.request
import tempfile
from io import StringIO

# Initialize scaler
scaler = StandardScaler()

# Web page settings
page_title = "💊 Breast Cancer pIC50 Prediction Web App"
page_icon = "🎗🧬⌬"
layout = "centered"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

# App title and logo
st.image('images/logo.png', use_column_width=True)

# Navigation menu
selected = option_menu(
    menu_title=page_title + " " + page_icon,
    options=['Home', 'Select Target', 'About', 'Contact'],
    icons=["house-fill", "capsule", "info-circle", "envelope-fill"],
    default_index=0,
    orientation="horizontal"
)

# Available targets with model URLs
TARGET_MODELS = {
    'AKT': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/main/models/AKT_catboost_regression_model.cbm",
    'ER': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/main/models/ER_catboost_regression_model.cbm",
    'ATM': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/ATM_catboost_regression_model.cbm",
    'CDK2': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/CDK2_catboost_regression_model.cbm",
    'CXCR4': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/CXCR4_catboost_regression_model.cbm",
    'FGFR': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/FGFR_catboost_regression_model.cbm",
    'VEGFR2': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/VEGFR2_catboost_regression_model.cbm",
    'BRAF': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/braf_catboost_regression_model.cbm",
    'GSK3B': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/GSK3B_catboost_regression_model.cbm",
    'HDAC1': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/HDAC1_catboost_regression_model.cbm",
    'HDAC2': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/HDAC2_catboost_regression_model.cbm",
    'HDAC8': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/HDAC8_catboost_regression_model.cbm",
    'JAK2': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/JAK2_catboost_regression_model.cbm",
    'PARP1': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/PARP1_catboost_regression_model.cbm",
    'aromatase': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/aromatase_catboost_regression_model.cbm",
    'mTOR': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/mTOR_catboost_regression_model.cbm",
    'PI3K': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/PI3K_catboost_regression_model.cbm",
    'PR': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/PR_catboost_regression_model.cbm",
    'HER2': "https://github.com/afolabiowoloye/Breast-Cancer-pIC50/raw/refs/heads/main/models/HER2_catboost_regression_model.cbm",
     # Add other targets here
}

targets = ['Click to select a target'] + sorted(TARGET_MODELS.keys())

# Getting RDKit descriptors with improved error handling
def RDKit_descriptors(SMILES):
    valid_smiles = []
    mols = []
    
    for i, smi in enumerate(SMILES):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            st.warning(f"Invalid SMILES at row {i+1}: {smi}")
            continue
        valid_smiles.append(smi)
        mols.append(mol)
    
    if not mols:
        return None, None, None
    
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors = []
    for mol in mols:
        try:
            mol = Chem.AddHs(mol)
            descriptors = calc.CalcDescriptors(mol)
            Mol_descriptors.append(descriptors)
        except Exception as e:
            st.warning(f"Failed to calculate descriptors for molecule: {str(e)}")
            continue
    
    return Mol_descriptors, desc_names, valid_smiles

# Home page
if selected == "Home":
    st.markdown("""
    <h3 style='color: darkblue;'>Welcome to Breast Cancer pIC<sub>50</sub> Prediction Web App</h3>
    """, unsafe_allow_html=True)
    
    st.image('images/workflow.png', use_column_width=True)

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

# Target selection page
if selected == "Select Target":
    st.subheader("Select preferred target")
    selected_target = st.selectbox("👇", targets)

    if selected_target == "Click to select a target":
        st.info("Please select a target protein from the dropdown list")
    else:
        # Improved model loading with caching and temp files
        @st.cache_resource(ttl=24*3600, show_spinner="Loading prediction model...")
        def load_model(target):
            MODEL_URL = TARGET_MODELS[target]
            
            # Create temp directory if it doesn't exist
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, f"{target}_model.cbm")
            
            if not os.path.exists(model_path):
                try:
                    urllib.request.urlretrieve(MODEL_URL, model_path)
                    st.success(f"{target} model downloaded successfully")
                except Exception as e:
                    st.error(f"Failed to download model: {str(e)}")
                    return None
            
            try:
                model = CatBoostRegressor()
                model.load_model(model_path)
                return model
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return None

        model = load_model(selected_target)
        
        if model is not None:
            # File upload section with example data
            st.subheader("Upload Compound Data")
            
            # Example data
            example_data = """name,SMILES
Example1,CC(=O)OC1=CC=CC=C1C(=O)O
Example2,C1=CC=C(C=C1)C=O"""
            
            # Show example and download button
            with st.expander("Show example input format"):
                st.code(example_data)
                st.download_button(
                    label="Download example CSV",
                    data=example_data,
                    file_name="example_compounds.csv",
                    mime="text/csv"
                )
            
            smiles_file = st.file_uploader(
                "Upload your compound data (CSV)",
                type=["csv"],
                help="File must contain 'SMILES' column and optionally 'name' column"
            )
            
            if smiles_file is not None:
                try:
                    # Read and validate input
                    sample = pd.read_csv(smiles_file)
                    
                    if 'SMILES' not in sample.columns:
                        st.error("Error: Input file must contain a 'SMILES' column")
                        st.stop()
                    
                    # Show preview
                    st.subheader("Input Data Preview")
                    st.dataframe(sample.head())
                    
                    # Calculate descriptors
                    with st.spinner("Calculating molecular descriptors..."):
                        descriptors, desc_names, valid_smiles = RDKit_descriptors(sample['SMILES'])
                    
                    if not descriptors:
                        st.error("Error: No valid molecules found in the input file")
                        st.stop()
                    
                    # Create descriptors dataframe
                    df_descriptors = pd.DataFrame(descriptors, columns=desc_names)
                    
                    # Remove problematic columns
                    problematic_cols = ["SPS", "AvgIpc"]
                    cols_to_drop = [col for col in problematic_cols if col in df_descriptors.columns]
                    if cols_to_drop:
                        df_descriptors = df_descriptors.drop(cols_to_drop, axis=1)
                    
                    # Make predictions
                    with st.spinner("Making predictions..."):
                        X_scaled = scaler.fit_transform(df_descriptors)
                        predictions = model.predict(X_scaled)
                    
                    # Prepare results
                    results = sample[sample['SMILES'].isin(valid_smiles)].copy()
                    results['predicted_pIC50'] = predictions
                    results = results.round({'predicted_pIC50': 3})  # Round to 3 decimal places
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(results[['name', 'SMILES', 'predicted_pIC50']])
                    
                    # Download results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results",
                        data=csv,
                        file_name=f"{selected_target}_predictions.csv",
                        mime="text/csv",
                        help="Download complete prediction results with all descriptors"
                    )
                    
                    # Show some statistics
                    st.subheader("Prediction Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Compounds Processed", len(results))
                    col2.metric("Average pIC50", f"{results['predicted_pIC50'].mean():.2f}")
                    col3.metric("Highest pIC50", f"{results['predicted_pIC50'].max():.2f}")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

    with st.sidebar.header("""Available Targets"""):
        st.sidebar.markdown("""
        <strong>1. Aromatase: </strong>Aromatase<br>
        <strong>2. JAK2: </strong>Janus Kinase 2<br>
        <strong>3. AKT: </strong>Protein Kinase B<br>
        <strong>4. ER: </strong>Estrogen Receptor<br>
        <strong>5. PR: </strong>Progesterone Receptor<br>
        <strong>6. BRAF: </strong>B-Raf Proto-Oncogene<br>
        <strong>7. HDAC1: </strong>Histone Deacetylase 1<br>
        <strong>8. HDAC2: </strong>Histone Deacetylase 2<br>
        <strong>9. HDAC8: </strong>Histone Deacetylase 8<br>
        <strong>10. CDK2: </strong>Cyclin-Dependent Kinase 2<br>
        <strong>11. PI3K: </strong>Phosphoinositide 3-Kinase<br>
        <strong>12. ATM: </strong>Ataxia Telangiectasia Mutated<br>
        <strong>13. mTOR: </strong>Mammalian Target of Rapamycin<br>
        <strong>14. PARP1: </strong>Poly(ADP-ribose) Polymerase 1<br>
        <strong>15. GSK3B: </strong>Glycogen Synthase Kinase 3 Beta<br>
        <strong>16. CXCR4: </strong>C-X-C Chemokine Receptor Type 4<br>
        <strong>17. FGFR: </strong>Fibroblast Growth Factor Receptor<br>
        <strong>18. HER2: </strong>Human Epidermal Growth Factor Receptor 2<br>
        <strong>19. VEGFR2: </strong>Vascular Endothelial Growth Factor Receptor 2<br>
        """, unsafe_allow_html=True)
        
