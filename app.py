import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        color: transparent;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        animation: gradientShift 20s ease infinite;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        letter-spacing: 2px;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.1) 25%, 
            rgba(240, 147, 251, 0.1) 50%, 
            rgba(245, 87, 108, 0.1) 75%, 
            rgba(79, 172, 254, 0.1) 100%);
        background-size: 300% 300%;
        animation: gradientShift 20s ease infinite;
        border-radius: 20px;
        border: 2px solid transparent;
        background-clip: padding-box;
        box-shadow: 
            0 8px 32px rgba(102, 126, 234, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        z-index: -1;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        text-align: center;
    }
    .metric-card h2 {
        color: #2c3e50 !important;
        font-weight: bold;
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    .insight-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box h4 {
        color: #1a5c1a !important;
        font-weight: bold;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Data loading and caching
@st.cache_data
def load_data():
    """Load and preprocess the diabetes dataset"""
    try:
        from ucimlrepo import fetch_ucirepo
        diabetes_dataset = fetch_ucirepo(id=296)
        X_df = diabetes_dataset.data.features
        y_df = diabetes_dataset.data.targets
        df = pd.concat([X_df, y_df], axis=1)
        
        # Handle missing values and special characters
        for col in df.select_dtypes(include=['object']).columns:
            if '?' in df[col].values:
                df[col] = df[col].replace('?', np.nan)
        
        # Add diagnosis grouping
        df = add_diagnosis_groups(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return sample data structure for demonstration
        sample_df = pd.DataFrame({
            'age': ['[50-60)', '[60-70)', '[70-80)'] * 100,
            'gender': ['Male', 'Female'] * 150,
            'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'] * 100,
            'time_in_hospital': np.random.randint(1, 15, 300),
            'num_medications': np.random.randint(1, 30, 300),
            'number_diagnoses': np.random.randint(1, 10, 300),
            'readmitted': ['NO', '<30', '>30'] * 100,
            'diag_1': ['250', '401', '428'] * 100,
            'diag_2': ['250.01', '414', '486'] * 100,
            'diag_3': ['250.02', '427', '599'] * 100
        })
        return add_diagnosis_groups(sample_df)

def map_icd9_to_group(code):
    """Map ICD-9 diagnosis codes to diagnostic groups"""
    if pd.isna(code) or code == '' or str(code).lower() == 'nan':
        return 'Unknown'
    
    # Convert to string and handle various formats
    code_str = str(code).strip()
    
    # Remove any trailing decimals like .0
    if '.' in code_str:
        try:
            # Handle decimal codes
            code_num = float(code_str)
        except:
            return 'Other'
    else:
        try:
            # Handle integer codes
            code_num = float(code_str)
        except:
            return 'Other'
    
    # Apply ICD-9 grouping rules
    if 250.0 <= code_num < 251.0:  # Diabetes: 250.xx
        return 'Diabetes'
    elif (390 <= code_num <= 459) or code_num == 785:  # Circulatory
        return 'Circulatory'
    elif (520 <= code_num <= 579) or code_num == 787:  # Digestive
        return 'Digestive'
    elif (580 <= code_num <= 629) or code_num == 788:  # Genitourinary
        return 'Genitourinary'
    elif 800 <= code_num <= 999:  # Injury
        return 'Injury'
    elif 710 <= code_num <= 739:  # Musculoskeletal
        return 'Musculoskeletal'
    elif 140 <= code_num <= 239:  # Neoplasms
        return 'Neoplasms'
    elif (460 <= code_num <= 519) or code_num == 786:  # Respiratory
        return 'Respiratory'
    else:
        return 'Other'

def add_diagnosis_groups(df):
    """Add diagnosis group columns to the dataframe"""
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    
    for col in diag_cols:
        if col in df.columns:
            group_col = f"{col}_group"
            df[group_col] = df[col].apply(map_icd9_to_group)
    
    return df

@st.cache_data
def get_filtered_data(df, filters):
    """Apply filters to the dataset"""
    filtered_df = df.copy()
    
    for column, values in filters.items():
        if values and column in filtered_df.columns:
            if isinstance(values, list) and len(values) > 0:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
            elif not isinstance(values, list) and values != "All":
                filtered_df = filtered_df[filtered_df[column] == values]
    
    return filtered_df

def create_metric_cards(filtered_df, target_col):
    """Create metric cards for key statistics using filtered data"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4; font-weight: bold;">👥 Total Patients</h3>
            <h2 style="color: #2c3e50; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{:,}</h2>
        </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    with col2:
        avg_stay = filtered_df['time_in_hospital'].mean() if 'time_in_hospital' in filtered_df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff7f0e; font-weight: bold;">⏱️ Avg Hospital Stay</h3>
            <h2 style="color: #2c3e50; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{:.1f} days</h2>
        </div>
        """.format(avg_stay), unsafe_allow_html=True)
    
    with col3:
        avg_meds = filtered_df['num_medications'].mean() if 'num_medications' in filtered_df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2ca02c; font-weight: bold;">💊 Avg Medications</h3>
            <h2 style="color: #2c3e50; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{:.1f}</h2>
        </div>
        """.format(avg_meds), unsafe_allow_html=True)
    
    with col4:
        if target_col in filtered_df.columns:
            readmit_rate = (filtered_df[target_col] == '<30').mean() * 100
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #d62728; font-weight: bold;">🔄 Readmission Rate</h3>
                <h2 style="color: #2c3e50; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{:.1f}%</h2>
            </div>
            """.format(readmit_rate), unsafe_allow_html=True)


def create_column_distributions(filtered_df):
    """Create comprehensive distribution analysis for all columns"""
    st.markdown('<div class="section-header">📊 Column Distributions Analysis</div>', unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.warning("No data available for distribution analysis with current filters")
        return
    
    # Separate numerical and categorical columns
    numerical_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove any columns that might cause issues
    exclude_cols = ['encounter_id', 'patient_nbr'] if any(col in filtered_df.columns for col in ['encounter_id', 'patient_nbr']) else []
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    st.subheader(f"📈 Numerical Columns ({len(numerical_cols)} columns)")
    
    if numerical_cols:
        # User can select which numerical columns to display
        selected_numerical = st.multiselect(
            "Select numerical columns to display:",
            numerical_cols,
            default=numerical_cols[:6] if len(numerical_cols) > 6 else numerical_cols
        )
        
        if selected_numerical:
            # Create histograms for numerical columns
            num_cols = min(3, len(selected_numerical))
            num_rows = (len(selected_numerical) + num_cols - 1) // num_cols
            
            for i in range(0, len(selected_numerical), num_cols):
                cols = st.columns(num_cols)
                for j, col_name in enumerate(selected_numerical[i:i+num_cols]):
                    with cols[j]:
                        if col_name in filtered_df.columns and not filtered_df[col_name].empty:
                            fig = px.histogram(
                                filtered_df,
                                x=col_name,
                                title=f"Distribution of {col_name}",
                                nbins=30,
                                marginal="box"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True, key=f"hist_numerical_{col_name}_{i}_{j}")
                            
                            # Show basic statistics
                            stats_data = filtered_df[col_name].describe()
                            st.write("**Statistics:**")
                            st.write(f"• Mean: {stats_data['mean']:.2f}")
                            st.write(f"• Median: {stats_data['50%']:.2f}")
                            st.write(f"• Std Dev: {stats_data['std']:.2f}")
                            st.write(f"• Min: {stats_data['min']:.2f}")
                            st.write(f"• Max: {stats_data['max']:.2f}")
    else:
        st.write("No numerical columns available for distribution analysis.")
    
    st.subheader(f"📊 Categorical Columns ({len(categorical_cols)} columns)")
    
    if categorical_cols:
        # User can select which categorical columns to display
        selected_categorical = st.multiselect(
            "Select categorical columns to display:",
            categorical_cols,
            default=categorical_cols[:6] if len(categorical_cols) > 6 else categorical_cols
        )
        
        if selected_categorical:
            # Create bar charts for categorical columns
            num_cols = min(2, len(selected_categorical))
            
            for i in range(0, len(selected_categorical), num_cols):
                cols = st.columns(num_cols)
                for j, col_name in enumerate(selected_categorical[i:i+num_cols]):
                    with cols[j]:
                        if col_name in filtered_df.columns and not filtered_df[col_name].empty:
                            value_counts = filtered_df[col_name].value_counts().head(15)  # Top 15 values
                            
                            if len(value_counts) > 0:
                                fig = px.bar(
                                    x=value_counts.values,
                                    y=value_counts.index,
                                    orientation='h',
                                    title=f"Distribution of {col_name}",
                                    labels={'x': 'Count', 'y': col_name}
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True, key=f"bar_categorical_{col_name}_{i}_{j}")
                                
                                # Show basic statistics
                                st.write("**Statistics:**")
                                st.write(f"• Unique values: {filtered_df[col_name].nunique()}")
                                st.write(f"• Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
                                st.write(f"• Missing values: {filtered_df[col_name].isnull().sum()}")
                                
                                # Show top values
                                if len(value_counts) > 1:
                                    st.write("**Top 5 values:**")
                                    for idx, (val, count) in enumerate(value_counts.head(5).items()):
                                        pct = (count / len(filtered_df)) * 100
                                        st.write(f"  {idx+1}. {val}: {count} ({pct:.1f}%)")
    else:
        st.write("No categorical columns available for distribution analysis.")
    
    # Summary statistics table
    st.subheader("📋 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if numerical_cols:
            st.write("**Numerical Columns Summary:**")
            summary_stats = filtered_df[numerical_cols].describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
    
    with col2:
        if categorical_cols:
            st.write("**Categorical Columns Summary:**")
            cat_summary = []
            for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
                mode_values = filtered_df[col].mode()
                most_common = mode_values[0] if len(mode_values) > 0 and not filtered_df[col].empty else 'N/A'
                cat_summary.append({
                    'Column': col,
                    'Unique Values': filtered_df[col].nunique(),
                    'Most Common': most_common,
                    'Missing Values': filtered_df[col].isnull().sum(),
                    'Missing %': f"{(filtered_df[col].isnull().sum() / len(filtered_df) * 100):.1f}%"
                })
            
            cat_summary_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_summary_df, use_container_width=True)


def create_demographic_medical_analysis(filtered_df):
    """Create comprehensive demographic and medical analysis using filtered data"""
    st.markdown('<div class="section-header">📊 Demographics & Medical Patterns by Diagnosis</div>', unsafe_allow_html=True)
    
    # Time in hospital analysis - USING FILTERED DATA
    if all(col in filtered_df.columns for col in ['time_in_hospital', 'age', 'diag_1_group']):
        st.subheader("🏥 Hospital Stay Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Age & Diagnosis Category (Filtered)**")
            # Use filtered_df for pivot table
            age_hospital_diag = filtered_df.pivot_table(values='time_in_hospital', 
                                                      index='age', 
                                                      columns='diag_1_group', 
                                                      aggfunc='mean').fillna(0)
            
            # Select top 5 diagnoses from filtered data
            top_diag = filtered_df['diag_1_group'].value_counts().head(5).index
            age_hospital_subset = age_hospital_diag[top_diag] if len(top_diag) > 0 else age_hospital_diag
            
            if not age_hospital_subset.empty:
                fig = px.imshow(age_hospital_subset.values,
                               x=age_hospital_subset.columns,
                               y=age_hospital_subset.index,
                               color_continuous_scale="YlOrRd",
                               title="Average Hospital Days by Age & Diagnosis")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="heatmap_age_hospital_diag")
            else:
                st.write("No data available for current filters")
        
        with col2:
            st.write("**By Age (Overall - Filtered)**")
            # Use filtered_df for groupby
            age_hospital = filtered_df.groupby('age')['time_in_hospital'].mean()
            
            if not age_hospital.empty:
                fig = px.bar(x=age_hospital.index, 
                            y=age_hospital.values,
                            title="Average Hospital Days by Age Group",
                            labels={'x': 'Age Group', 'y': 'Average Days'},
                            color=age_hospital.values,
                            color_continuous_scale="Blues")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="bar_age_hospital")
            else:
                st.write("No data available for current filters")
    
    # Medications analysis - USING FILTERED DATA
    if all(col in filtered_df.columns for col in ['num_medications', 'gender', 'diag_1_group']):
        st.subheader("💊 Medication Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Gender & Diagnosis Category (Filtered)**")
            # Use filtered_df for pivot table
            gender_med_diag = filtered_df.pivot_table(values='num_medications', 
                                                    index='gender', 
                                                    columns='diag_1_group', 
                                                    aggfunc='mean').fillna(0)
            
            top_diag_meds = filtered_df['diag_1_group'].value_counts().head(6).index
            gender_med_subset = gender_med_diag[top_diag_meds] if len(top_diag_meds) > 0 else gender_med_diag
            
            if not gender_med_subset.empty:
                fig = px.bar(gender_med_subset,
                            title="Average Medications by Gender & Diagnosis",
                            labels={'value': 'Average Medications', 'index': 'Gender'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="bar_gender_med_diag")
            else:
                st.write("No data available for current filters")
        
        with col2:
            st.write("**By Gender (Overall - Filtered)**")
            # Use filtered_df for groupby
            gender_med = filtered_df.groupby('gender')['num_medications'].mean()
            
            if not gender_med.empty:
                fig = px.bar(x=gender_med.index,
                            y=gender_med.values,
                            title="Average Medications by Gender",
                            labels={'x': 'Gender', 'y': 'Average Medications'},
                            color=gender_med.values,
                            color_continuous_scale="Greens")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="bar_gender_med")
            else:
                st.write("No data available for current filters")
    
    # Number of diagnoses analysis - USING FILTERED DATA
    if all(col in filtered_df.columns for col in ['number_diagnoses', 'race', 'diag_1_group']):
        st.subheader("🔬 Diagnosis Count Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Race & Primary Diagnosis (Filtered)**")
            # Focus on top races from filtered data
            top_races = filtered_df['race'].value_counts().head(5).index
            df_race_subset = filtered_df[filtered_df['race'].isin(top_races)]
            
            if not df_race_subset.empty:
                race_numdiag_diag = df_race_subset.pivot_table(values='number_diagnoses', 
                                                             index='race', 
                                                             columns='diag_1_group', 
                                                             aggfunc='mean').fillna(0)
                
                top_diag_race = filtered_df['diag_1_group'].value_counts().head(4).index
                race_numdiag_subset = race_numdiag_diag[top_diag_race] if len(top_diag_race) > 0 else race_numdiag_diag
                
                if not race_numdiag_subset.empty:
                    fig = px.imshow(race_numdiag_subset.values,
                                   x=race_numdiag_subset.columns,
                                   y=race_numdiag_subset.index,
                                   color_continuous_scale="Blues",
                                   title="Average # Diagnoses by Race & Primary Diagnosis")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="heatmap_race_numdiag")
                else:
                    st.write("No data available for current filters")
            else:
                st.write("No data available for current filters")
        
        with col2:
            st.write("**By Race (Overall - Filtered)**")
            top_races = filtered_df['race'].value_counts().head(8).index
            race_numdiag = filtered_df[filtered_df['race'].isin(top_races)].groupby('race')['number_diagnoses'].mean()
            
            if not race_numdiag.empty:
                fig = px.bar(x=race_numdiag.values,
                            y=race_numdiag.index,
                            orientation='h',
                            title="Average # Diagnoses by Race",
                            labels={'x': 'Average # Diagnoses', 'y': 'Race'},
                            color=race_numdiag.values,
                            color_continuous_scale="Oranges")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="bar_race_numdiag")
            else:
                st.write("No data available for current filters")
    
    # Box plots for distributions - USING FILTERED DATA
    if all(col in filtered_df.columns for col in ['time_in_hospital', 'race']):
        st.subheader("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hospital Stay Distribution by Race (Filtered)**")
            top_races = filtered_df['race'].value_counts().head(6).index
            df_race_subset = filtered_df[filtered_df['race'].isin(top_races)]
            
            if not df_race_subset.empty and len(df_race_subset) > 10:
                fig = px.box(df_race_subset, 
                            x='race', 
                            y='time_in_hospital',
                            title="Hospital Stay Distribution by Race")
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="box_race_hospital")
            else:
                st.write("Insufficient data for current filters")
        
        with col2:
            if 'num_medications' in filtered_df.columns and 'age' in filtered_df.columns:
                st.write("**Medications Distribution by Age (Filtered)**")
                key_ages = ['[50-60)', '[60-70)', '[70-80)', '[80-90)']
                available_ages = [age for age in key_ages if age in filtered_df['age'].unique()]
                df_age_subset = filtered_df[filtered_df['age'].isin(available_ages)]
                
                if not df_age_subset.empty and len(df_age_subset) > 10:
                    fig = px.box(df_age_subset,
                                x='age',
                                y='num_medications',
                                title="Medications Distribution by Age Group")
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="box_age_medications")
                else:
                    st.write("Insufficient data for current filters")

def create_diagnosis_analysis(filtered_df):
    """Create comprehensive diagnosis analysis using filtered data"""
    st.markdown('<div class="section-header">🏥 Diagnosis Pattern Analysis</div>', unsafe_allow_html=True)
    
    # Check if diagnosis group columns exist
    diag_group_cols = [col for col in filtered_df.columns if col.endswith('_group')]
    
    if not diag_group_cols:
        st.warning("No diagnosis group data available")
        return
    
    # Overview of diagnosis groups - USING FILTERED DATA
    st.subheader("Diagnosis Categories Overview (Current Filters)")
    
    # Primary diagnosis analysis - USING FILTERED DATA
    if 'diag_1_group' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Primary diagnosis distribution from filtered data
            primary_diag = filtered_df['diag_1_group'].value_counts()
            
            if not primary_diag.empty:
                fig = px.pie(
                    values=primary_diag.values,
                    names=primary_diag.index,
                    title="Primary Diagnosis Distribution (Filtered)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="pie_primary_diagnosis")
                
                # Show statistics from filtered data
                st.write("**Primary Diagnosis Statistics (Filtered):**")
                for diag, count in primary_diag.head(5).items():
                    pct = (count / len(filtered_df)) * 100
                    st.write(f"• {diag}: {count:,} patients ({pct:.1f}%)")
            else:
                st.write("No diagnosis data available for current filters")
        
        with col2:
            # Primary diagnosis bar chart from filtered data
            if not primary_diag.empty:
                fig = px.bar(
                    x=primary_diag.values,
                    y=primary_diag.index,
                    orientation='h',
                    title="Primary Diagnosis Counts (Filtered)",
                    labels={'x': 'Number of Patients', 'y': 'Diagnosis Category'},
                    color=primary_diag.values,
                    color_continuous_scale="viridis"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="bar_primary_diagnosis")
    
    # Compare all diagnosis positions - USING FILTERED DATA
    st.subheader("Diagnosis Patterns Across Positions (Current Filters)")
    
    # Create comparison data from filtered data
    if all(col in filtered_df.columns for col in ['diag_1_group', 'diag_2_group', 'diag_3_group']):
        diag_comparison = pd.DataFrame({
            'Primary (diag_1)': filtered_df['diag_1_group'].value_counts(),
            'Secondary (diag_2)': filtered_df['diag_2_group'].value_counts(),
            'Additional (diag_3)': filtered_df['diag_3_group'].value_counts()
        }).fillna(0)
        
        if not diag_comparison.empty:
            # Stacked bar chart
            fig = px.bar(
                diag_comparison,
                title="Diagnosis Categories by Position (Filtered)",
                labels={'value': 'Number of Patients', 'index': 'Diagnosis Category'},
                barmode='group',
                height=500
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="bar_diagnosis_comparison")
        else:
            st.write("No diagnosis comparison data available for current filters")

def create_diagnosis_readmission_analysis(filtered_df, target_col):
    """Analyze diagnosis patterns by readmission status using filtered data"""
    st.markdown('<div class="section-header">🔄 Diagnosis vs Readmission Analysis</div>', unsafe_allow_html=True)
    
    if target_col not in filtered_df.columns or 'diag_1_group' not in filtered_df.columns:
        st.warning("Cannot perform diagnosis-readmission analysis: missing required columns")
        return
    
    # Readmission rates by primary diagnosis - USING FILTERED DATA
    st.subheader("Readmission Risk by Diagnosis Category (Current Filters)")
    
    # Calculate readmission rates by diagnosis group from filtered data
    diag_readmit = pd.crosstab(filtered_df['diag_1_group'], filtered_df[target_col], normalize='index') * 100
    
    if not diag_readmit.empty:
        # Create stacked bar chart
        fig = px.bar(
            diag_readmit,
            title="Readmission Rates by Primary Diagnosis Category (%) - Filtered",
            labels={'value': 'Percentage of Patients', 'index': 'Diagnosis Category'},
            height=500
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key="bar_readmission_diagnosis")
    
    # Calculate overall readmission rate FIRST, outside of any column blocks
    overall_readmit_rate = 0
    if target_col in filtered_df.columns and len(filtered_df) > 0:
        overall_readmit_rate = (filtered_df[target_col] == '<30').mean() * 100
    
    # 1. Readmission by Age Group
    if 'age' in filtered_df.columns:
        st.subheader("📊 Readmission Rate by Age Group")
        
        age_readmission = pd.crosstab(filtered_df['age'], filtered_df[target_col], normalize='index') * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                age_readmission,
                title="Readmission Rate by Age Group (%)",
                labels={'value': 'Percentage', 'index': 'Age Group'},
                height=400
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="bar_age_readmission")
        
        with col2:
            # Show statistics table
            st.write("**Readmission Rates by Age Group:**")
            age_stats = age_readmission.round(1)
            st.dataframe(age_stats, use_container_width=True)
    
    # 2. Readmission by Gender
    if 'gender' in filtered_df.columns:
        st.subheader("👥 Readmission Rate by Gender")
        
        gender_readmission = pd.crosstab(filtered_df['gender'], filtered_df[target_col], normalize='index') * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                gender_readmission,
                title="Readmission Rate by Gender (%)",
                labels={'value': 'Percentage', 'index': 'Gender'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key="bar_gender_readmission")
        
        with col2:
            # Show statistics table
            st.write("**Readmission Rates by Gender:**")
            gender_stats = gender_readmission.round(1)
            st.dataframe(gender_stats, use_container_width=True)
    
    # 3. Readmission by Hospital Stay Duration
    if 'time_in_hospital' in filtered_df.columns:
        st.subheader("🏥 Hospital Stay Duration by Readmission Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram overlayed by readmission status
            fig = px.histogram(
                filtered_df,
                x='time_in_hospital',
                color=target_col,
                title='Hospital Stay Duration by Readmission Status',
                labels={'time_in_hospital': 'Time in Hospital (days)', 'count': 'Frequency'},
                opacity=0.6,
                nbins=15,
                barmode='overlay'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="hist_hospital_readmission")
        
        with col2:
            # Box plot
            fig = px.box(
                filtered_df,
                x=target_col,
                y='time_in_hospital',
                title='Hospital Stay Duration Distribution by Readmission Status',
                labels={'time_in_hospital': 'Days in Hospital'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="box_hospital_readmission")
    
    # 4. Diagnosis-specific readmission analysis
    if 'diag_1_group' in filtered_df.columns:
        st.subheader("🔬 Readmission Risk by Diagnosis Category")
        
        # Statistical analysis - USING FILTERED DATA
        col1, col2 = st.columns(2)

        st.write(f"**Overall readmission rate (filtered):** {overall_readmit_rate:.1f}%")
        st.write("")
        
        with col1:
            st.subheader("High-Risk Diagnosis Categories (Filtered)")
            
            # Calculate overall readmission rate for comparison from filtered data
            if target_col in filtered_df.columns:

                # Find diagnosis groups with higher than average readmission from filtered data
                high_risk_diagnoses = []
                for diag_group in filtered_df['diag_1_group'].unique():
                    if pd.notna(diag_group) and diag_group != 'Unknown':
                        subset = filtered_df[filtered_df['diag_1_group'] == diag_group]
                        diag_readmit_rate = (subset[target_col] == '<30').mean() * 100
                        patient_count = len(subset)
                        
                        if diag_readmit_rate > overall_readmit_rate and patient_count >= 5:  # Reduced minimum sample size for filtered data
                            high_risk_diagnoses.append({
                                'Diagnosis': diag_group,
                                'Readmission Rate (%)': round(diag_readmit_rate, 1),
                                'Patients': patient_count,
                                'Risk Ratio': round(diag_readmit_rate / overall_readmit_rate, 2) if overall_readmit_rate > 0 else 1
                            })
                
                high_risk_diagnoses_2 = []
                for diag_group in filtered_df['diag_2_group'].unique():
                    if pd.notna(diag_group) and diag_group != 'Unknown':
                        subset = filtered_df[filtered_df['diag_2_group'] == diag_group]
                        diag_readmit_rate = (subset[target_col] == '<30').mean() * 100
                        patient_count = len(subset)
                        
                        if diag_readmit_rate > overall_readmit_rate and patient_count >= 5:  # Reduced minimum sample size for filtered data
                            high_risk_diagnoses_2.append({
                                'Diagnosis': diag_group,
                                'Readmission Rate (%)': round(diag_readmit_rate, 1),
                                'Patients': patient_count,
                                'Risk Ratio': round(diag_readmit_rate / overall_readmit_rate, 2) if overall_readmit_rate > 0 else 1
                            })

                # Sort by risk ratio
                high_risk_diagnoses = sorted(high_risk_diagnoses, key=lambda x: x['Risk Ratio'], reverse=True)
                
                if high_risk_diagnoses:
                    st.write("**Above-average readmission risk for primary diagnosis (in current filters):**")
                    for risk_diag in high_risk_diagnoses[:5]:
                        st.write(f"🔴 **{risk_diag['Diagnosis']}**")
                        st.write(f"   Rate: {risk_diag['Readmission Rate (%)']}% | Patients: {risk_diag['Patients']:,} | Risk: {risk_diag['Risk Ratio']}x")
                else:
                    st.write("No diagnosis categories show significantly elevated readmission risk in current filters.")

                high_risk_diagnoses_2 = sorted(high_risk_diagnoses_2, key=lambda x: x['Risk Ratio'], reverse=True)

                if high_risk_diagnoses_2:
                    st.write("**Above-average readmission risk for secondary diagnosis (in current filters):**")
                    for risk_diag in high_risk_diagnoses_2[:5]:
                        st.write(f"🔴 **{risk_diag['Diagnosis']}**")
                        st.write(f"   Rate: {risk_diag['Readmission Rate (%)']}% | Patients: {risk_diag['Patients']:,} | Risk: {risk_diag['Risk Ratio']}x")
                else:
                    st.write("No diagnosis categories show significantly elevated readmission risk in current filters.")
        
        with col2:
            st.subheader("Low-Risk Diagnosis Categories (Filtered)")
            
            # Find diagnosis groups with lower than average readmission from filtered data
            low_risk_diagnoses = []
            for diag_group in filtered_df['diag_1_group'].unique():
                if pd.notna(diag_group) and diag_group != 'Unknown':
                    subset = filtered_df[filtered_df['diag_1_group'] == diag_group]
                    diag_readmit_rate = (subset[target_col] == '<30').mean() * 100
                    patient_count = len(subset)
                    
                    if diag_readmit_rate < overall_readmit_rate and patient_count >= 5:  # Reduced minimum sample size for filtered data
                        low_risk_diagnoses.append({
                            'Diagnosis': diag_group,
                            'Readmission Rate (%)': round(diag_readmit_rate, 1),
                            'Patients': patient_count,
                            'Risk Ratio': round(diag_readmit_rate / overall_readmit_rate, 2) if overall_readmit_rate > 0 else 1
                        })

            low_risk_diagnoses_2 = []
            for diag_group in filtered_df['diag_2_group'].unique():
                if pd.notna(diag_group) and diag_group != 'Unknown':
                    subset = filtered_df[filtered_df['diag_2_group'] == diag_group]
                    diag_readmit_rate = (subset[target_col] == '<30').mean() * 100
                    patient_count = len(subset)
                    
                    if diag_readmit_rate < overall_readmit_rate and patient_count >= 5:  # Reduced minimum sample size for filtered data
                        low_risk_diagnoses_2.append({
                            'Diagnosis': diag_group,
                            'Readmission Rate (%)': round(diag_readmit_rate, 1),
                            'Patients': patient_count,
                            'Risk Ratio': round(diag_readmit_rate / overall_readmit_rate, 2) if overall_readmit_rate > 0 else 1
                        })
            
            # Sort by lowest risk ratio
            low_risk_diagnoses = sorted(low_risk_diagnoses, key=lambda x: x['Risk Ratio'])
            
            if low_risk_diagnoses:
                st.write("**Below-average readmission risk for primary diagnosis (in current filters):**")
                for risk_diag in low_risk_diagnoses[:5]:
                    st.write(f"🟢 **{risk_diag['Diagnosis']}**")
                    st.write(f"   Rate: {risk_diag['Readmission Rate (%)']}% | Patients: {risk_diag['Patients']:,} | Risk: {risk_diag['Risk Ratio']}x")
            else:
                st.write("All diagnosis categories show average or elevated readmission risk in current filters.")

            low_risk_diagnoses_2 = sorted(low_risk_diagnoses_2, key=lambda x: x['Risk Ratio'])

            if low_risk_diagnoses_2:
                st.write("**Below-average readmission risk for secondary diagnosis (in current filters):**")
                for risk_diag in low_risk_diagnoses_2[:5]:
                    st.write(f"🟢 **{risk_diag['Diagnosis']}**")
                    st.write(f"   Rate: {risk_diag['Readmission Rate (%)']}% | Patients: {risk_diag['Patients']:,} | Risk: {risk_diag['Risk Ratio']}x")
            else:
                st.write("All diagnosis categories show average or elevated readmission risk in current filters.")

def create_correlation_analysis(filtered_df):
    """Create correlation analysis using filtered data"""
    st.markdown('<div class="section-header">📊 Correlation Analysis</div>', unsafe_allow_html=True)
    
    # Get numerical columns from filtered data
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Let user select columns for correlation
        selected_cols = st.multiselect(
            "Select columns for correlation analysis:",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        )
        
        if len(selected_cols) > 1:
            # Use filtered data for correlation
            correlation_matrix = filtered_df[selected_cols].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                correlation_matrix,
                title="Correlation Heatmap (Filtered Data)",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
            
            # Show highly correlated pairs from filtered data
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        high_corr_pairs.append({
                            'Variable 1': correlation_matrix.columns[i],
                            'Variable 2': correlation_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if high_corr_pairs:
                st.subheader("Highly Correlated Pairs in Filtered Data (|correlation| > 0.5)")
                high_corr_df = pd.DataFrame(high_corr_pairs)
                high_corr_df['Correlation'] = high_corr_df['Correlation'].round(3)
                st.dataframe(high_corr_df, use_container_width=True)
            else:
                st.write("No highly correlated pairs found in filtered data.")
    else:
        st.write("Insufficient numerical columns for correlation analysis in filtered data.")

def create_insights_summary(filtered_df, target_col):
    """Create insights and summary section using filtered data"""
    st.markdown('<div class="section-header">💡 Key Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Key insights from filtered data
    insights = []
    
    # Demographics insights from filtered data
    if 'age' in filtered_df.columns and not filtered_df['age'].empty:
        mode_values_age = filtered_df['age'].mode()
        most_common_age = mode_values_age[0] if len(mode_values_age) > 0 and not filtered_df['age'].empty else 'N/A'
        insights.append(f"Most common age group (filtered): **{most_common_age}**")
    
    if 'gender' in filtered_df.columns and not filtered_df['gender'].empty:
        gender_dist = filtered_df['gender'].value_counts()
        if len(gender_dist) > 0:
            dominant_gender = gender_dist.index[0]
            gender_pct = gender_dist.iloc[0] / len(filtered_df) * 100
            insights.append(f"Gender distribution (filtered): **{dominant_gender}** dominates ({gender_pct:.1f}%)")
    
    # Diagnosis insights from filtered data
    if 'diag_1_group' in filtered_df.columns and not filtered_df['diag_1_group'].empty:
        mode_values_diag = filtered_df['diag_1_group'].mode()
        most_common_diag = mode_values_diag[0] if len(mode_values_diag) > 0 and not filtered_df['diag_1_group'].empty else 'N/A'
        diag_count = (filtered_df['diag_1_group'] == most_common_diag).sum()
        diag_pct = diag_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        insights.append(f"Most common primary diagnosis (filtered): **{most_common_diag}** ({diag_pct:.1f}%)")

    if 'diag_2_group' in filtered_df.columns and not filtered_df['diag_2_group'].empty:
        mode_values_diag2 = filtered_df['diag_2_group'].mode()
        most_common_diag2 = mode_values_diag2[0] if len(mode_values_diag2) > 0 and not filtered_df['diag_2_group'].empty else 'N/A'
        diag_count = (filtered_df['diag_2_group'] == most_common_diag2).sum()
        diag_pct = diag_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        insights.append(f"Most common secondary diagnosis (filtered): **{most_common_diag}** ({diag_pct:.1f}%)")

    # Medical insights from filtered data
    if 'time_in_hospital' in filtered_df.columns and not filtered_df['time_in_hospital'].empty:
        avg_stay = filtered_df['time_in_hospital'].mean()
        insights.append(f"Average hospital stay (filtered): **{avg_stay:.1f} days**")
    
    if 'num_medications' in filtered_df.columns and not filtered_df['num_medications'].empty:
        avg_medications = filtered_df['num_medications'].mean()
        insights.append(f"Average medications per patient (filtered): **{avg_medications:.1f}**")
    
    # Readmission insights from filtered data
    if target_col in filtered_df.columns and not filtered_df[target_col].empty:
        readmit_counts = filtered_df[target_col].value_counts(normalize=True) * 100
        for status, rate in readmit_counts.head(3).items():
            insights.append(f"Readmission - {status} (filtered): **{rate:.1f}%** of patients")
    
    # Diagnosis-specific readmission insights from filtered data
    if target_col in filtered_df.columns and 'diag_1_group' in filtered_df.columns and len(filtered_df) > 0:
        overall_readmit_rate = (filtered_df[target_col] == '<30').mean() * 100
        
        # Find highest risk diagnosis from filtered data
        highest_risk_diag = None
        highest_risk_rate = 0
        
        for diag_group in filtered_df['diag_1_group'].unique():
            if pd.notna(diag_group) and diag_group != 'Unknown':
                subset = filtered_df[filtered_df['diag_1_group'] == diag_group]
                if len(subset) >= 5:  # Reduced minimum sample size for filtered data
                    diag_readmit_rate = (subset[target_col] == '<30').mean() * 100
                    if diag_readmit_rate > highest_risk_rate:
                        highest_risk_rate = diag_readmit_rate
                        highest_risk_diag = diag_group
        
        if highest_risk_diag and overall_readmit_rate > 0:
            risk_ratio = highest_risk_rate / overall_readmit_rate
            insights.append(f"Highest readmission risk (filtered): **{highest_risk_diag}** ({highest_risk_rate:.1f}%, {risk_ratio:.1f}x average)")
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4 style="color: #1a5c1a; font-weight: bold; font-size: 1.2rem;">🔍 Key Dataset Insights (Filtered)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if insights:
            for insight in insights:
                st.write(f"• {insight}")
        else:
            st.write("• No insights available for current filters")
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4 style="color: #1a5c1a; font-weight: bold; font-size: 1.2rem;">📈 Clinical Recommendations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = [
            "Remove features with high missing rate",
            "Remove constant features",
            "Remove unneeded features",
            "Balance the dataset to address class imbalance",
            "Feature engineering to create new relevant features",
            "Monitor outlier cases for quality improvement" 
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Diabetes Hospital Readmission Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading diabetes dataset..."):
        df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Data Filters")
    
    filters = {}
    
    # Age filter
    if 'age' in df.columns:
        age_options = ['All'] + sorted(df['age'].dropna().unique().tolist())
        filters['age'] = st.sidebar.multiselect("Age Group", age_options, default=['All'])
        if 'All' in filters['age'] or not filters['age']:
            filters['age'] = []
    
    # Gender filter
    if 'gender' in df.columns:
        gender_options = ['All'] + df['gender'].dropna().unique().tolist()
        filters['gender'] = st.sidebar.selectbox("Gender", gender_options)
        if filters['gender'] == 'All':
            filters['gender'] = []
    
    # Race filter
    if 'race' in df.columns:
        race_options = ['All'] + sorted(df['race'].dropna().unique().tolist())
        selected_races = st.sidebar.multiselect("Race", race_options, default=['All'])
        if 'All' in selected_races or not selected_races:
            filters['race'] = []
        else:
            filters['race'] = selected_races
    
    # Primary diagnosis filter
    if 'diag_1_group' in df.columns:
        diag_options = ['All'] + sorted([d for d in df['diag_1_group'].dropna().unique().tolist() if d != 'Unknown'])
        selected_diags = st.sidebar.multiselect("Primary Diagnosis Category", diag_options, default=['All'])
        if 'All' in selected_diags or not selected_diags:
            filters['diag_1_group'] = []
        else:
            filters['diag_1_group'] = selected_diags
    
    # Readmission status filter
    if 'readmitted' in df.columns:
        readmit_options = ['All'] + df['readmitted'].dropna().unique().tolist()
        filters['readmitted'] = st.sidebar.selectbox("Readmission Status", readmit_options)
        if filters['readmitted'] == 'All':
            filters['readmitted'] = []
    
    # Apply filters to get filtered data
    filtered_df = get_filtered_data(df, filters)
    
    # Show filter results
    if len(filtered_df) != len(df):
        st.info(f"🔍 Showing {len(filtered_df):,} of {len(df):,} records after filtering")
    else:
        st.success(f"📊 Showing all {len(df):,} records (no filters applied)")
    
    # Target column detection
    target_col = 'readmitted'
    if target_col not in filtered_df.columns:
        possible_targets = [col for col in filtered_df.columns if 'readmit' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
    
    # Check if filtered data is empty
    if len(filtered_df) == 0:
        st.error("⚠️ No data available for the current filter selection. Please adjust your filters.")
        return

    # Missing data calculation
    missing_data = filtered_df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Percentage', ascending=False)
    missing_cols = missing_df[missing_df['Missing Count'] > 0]

    # Main dashboard content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "📈 All Distributions",
        "👥 Demographics & Medical",
        "🔬 Diagnosis Patterns",
        "🔄 Readmission Analysis",
        "📊 Correlations",
        "💡 Insights"
    ])
    
    with tab1:
        st.subheader("Dataset Overview")
        create_metric_cards(filtered_df, target_col)
        
        # Data preview
        st.subheader("Data Preview (Current Filters)")
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Missing Data
        st.subheader("Missing Value Analysis (Current Filters)")
        st.dataframe(missing_cols.head(10), use_container_width=True)

        # Constant Data
        st.subheader("Constant Value Analysis (Current Filters)")
        constant_data = filtered_df.nunique()
        constant_cols = constant_data[constant_data == 1].index.tolist()
        if constant_cols:
            st.dataframe(filtered_df[constant_cols].head(10), use_container_width=True)
        else:
            st.write("No constant columns found.")

        # Filter summary
        if any(filters.values()):
            st.subheader("Active Filters")
            for filter_name, filter_values in filters.items():
                if filter_values:
                    if isinstance(filter_values, list):
                        st.write(f"• **{filter_name.replace('_', ' ').title()}**: {', '.join(map(str, filter_values))}")
                    else:
                        st.write(f"• **{filter_name.replace('_', ' ').title()}**: {filter_values}")
    
    with tab2:
        create_column_distributions(filtered_df)
    
    with tab3:
        create_demographic_medical_analysis(filtered_df)
    
    with tab4:
        create_diagnosis_analysis(filtered_df)
    
    with tab5:
        create_diagnosis_readmission_analysis(filtered_df, target_col)
    
    with tab6:
        create_correlation_analysis(filtered_df)
    
    with tab7:
        create_insights_summary(filtered_df, target_col)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Dashboard Info:** Interactive analysis of Diabetes 130-US Hospitals dataset with ICD-9 diagnosis grouping | **Current View:** {len(filtered_df):,} patients")
    
    # Download filtered data option
    if st.button("📥 Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"diabetes_filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()