# DA231o-Detecting-Suspicious-Cryptocurrency-Transactions

## Project Setup
To start the project, run the `setup.py` script. This will start Apache Airflow and MLflow, and initiate a DAG that runs the stages in parallel.

## Problem Definition
Develop a scalable system to detect potentially suspicious or fraudulent cryptocurrency transactions using machine learning and big data techniques.

## Problem Motivation
Cryptocurrency fraud and illicit activities have become increasingly sophisticated, making it challenging for traditional rule-based systems to detect suspicious behaviour. There's an urgent need for an intelligent, adaptive system that can analyse large volumes of transaction data in real-time to identify potential threats.

## Design Goals & Features Supported
1. Batch processing of large-scale historical transaction data
2. Robust preprocessing pipeline for the large Bitcoin transaction dataset
3. Advanced feature extraction from transaction graphs and metadata
4. Statistical analysis to identify significant patterns and anomalies
5. Machine learning models for ransomware address classification
6. Predictive modelling for ransomware behaviour forecasting

## Approach
### High-level Design
1. **Data Preprocessing Module**
   - Data cleaning and standardisation
   - Feature extraction and engineering
   - Dataset partitioning for training and testing
2. **Analysis Pipeline**
   - Statistical analysis of transaction patterns
   - Machine learning models for classification and prediction
   - Ensemble methods for improved accuracy
3. **Visualisation and Reporting**
   - Network visualisation of transaction patterns
   - Statistical summaries and insights
   - Model performance analysis and visualisation

## Big Data Platforms Used
- Apache Spark for distributed computing
- Pyspark ML for machine learning
- MLflow for MLOps
- NetworkX for graph analysis
- Apache Airflow for workflow orchestration
- seaborn for interactive visualisations

## Data Sources and Data Models
### Dataset Characteristics:
- 2,916,697 Bitcoin transactions
- Features include: transaction IDs, timestamps, amounts, sender/receiver addresses, graph neighbours, length, weights, loops
- Labelled data for known ransomware addresses