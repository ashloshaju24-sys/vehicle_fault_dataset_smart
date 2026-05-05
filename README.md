# Comparative Analysis of Anomaly Detection Methods for Real-Time Engine and OBD-II Vehicle Signal Monitoring

**Run the code interactively in Google Colab:**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T2A5sKQsr09T1lzuEtG1QghjB4usXbd_?usp=sharing)

## Overview
This repository contains the implementation and resources for the 3rd Semester Research Work titled **"Comparative Analysis of Anomaly Detection Methods for Real-Time Engine and OBD-II Vehicle Signal Monitoring"**. The primary objective is to detect anomalies in real-time before catastrophic vehicle failures occur, leveraging machine learning and deep learning algorithms on continuous multi-dimensional time-series OBD-II sensor data.

## Motivation & Objectives
- **Problem:** Vehicle failures cost millions in maintenance and cause major safety risks.
- **Solution:** Detect anomalies in real-time before failure via predictive algorithms.
- **Impact:** Predictive maintenance can save ~30% in maintenance costs and prevent major safety incidents.

## Dataset
**Vehicle Fault Dataset (Smart)**
-**Dataset Source:** The data set was genrated sythetically using webscraping methods .
- **Total Records:** 30,000 multi-dimensional time-series samples.
- **Class Distribution:** 66.6% Normal (19,980) vs. 33.4% Anomaly (10,020).
- **Features:** 6 core OBD-II sensor signals (e.g., Engine RPM, Coolant Temperature, Oil Pressure).

## Models & Algorithms Evaluated
The research implements and compares the following algorithms:
1. **Isolation Forest:** A tree-based partitioning algorithm. Fastest performing, best for real-time edge computing.
2. **Local Outlier Factor (LOF):** Density-based algorithm identifying anomalies by comparing local point density. 
3. **One-Class SVM:** Kernel-based approach mapping normal data into an isolated dimension boundary.
4. **Autoencoder (Deep Learning):** Compression to a bottleneck layer, detecting anomalies via high reconstruction errors.

## Performance Comparison Baseline
| Algorithm | Precision | Recall | F1-Score | ROC-AUC | Time (s) |
|-----------|-----------|--------|----------|---------|----------|
| **Isolation Forest** | 71.2% | 68.9% | 70.0% | 0.8123 | 0.34 |
| **LOF** | 69.8% | 71.2% | 70.5% | 0.8045 | 2.15 |
| **One-Class SVM** | 70.3% | 69.7% | 70.0% | 0.8089 | 1.87 |
| **Autoencoder** | 72.5% | 75.8% | 74.1% | 0.8267 | 12.43 |

## Workflow & Feature Engineering
- **Exploratory Data Analysis (EDA):** Analysed class imbalances, engine signal dependencies (Pearson correlations), and multi-dimensional cluster distributions.
- **Preprocessing:** Sliding-window feature engineering capturing rolling means and standard deviations (5-interval window) to capture signal volatility over time.

## Evaluation
- Models were evaluated using Recall, Precision, F1-score, and ROC-AUC. Autoencoders showcased the highest discriminatory capabilities globally, while Isolation Forest presented excellent low-latency detection capabilities suitable for streaming real-time applications.

## Key Files
- `3rdAshlo_Shaju_Thesis_Code RNEW.ipynb`: Notebook containing code for dataset EDA, model processing, evaluation, and visualizations.
- `vehicle_fault_dataset_smart.csv`: Base dataset.
- `PPT_SLIDE_CONTENT_3RD_RESEARCH.md`: Core components outlining the thesis findings.
