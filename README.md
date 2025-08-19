# La Liga Match Result Prediction Project

This project aims to **predict match outcomes for the La Liga 2025-26 season** using historical data from 2019-24. The goal is to compare the performance of three machine learning algorithms—**LightGBM, XGBoost, and Random Forest**—and implement a system that automatically updates predictions and retrains after each match day.

***

## Project Overview

- **Objective:** Predict outcomes (win/draw/loss) for every fixture in the 2025-26 La Liga season, achieving at least 50% accuracy.
- **Data Used:**
  - `matches_full.xlsx` – Historical match records (2019-2024)
  - `la-liga-2025-UTC.xlsx` – Fixture list for the 2025-26 season
- **Algorithms Compared:**
  - LightGBM
  - XGBoost
  - Random Forest

***

# App Link : https://la-liga-score-prediction-sujoydas.streamlit.app/

## Features

- Cell-by-cell code structure compatible with **Google Colab**—no user-defined functions.
- Automated pipeline for:
  - Data preprocessing and feature engineering
  - Model training and evaluation
  - Prediction for upcoming fixtures

***

## Usage

1. **Clone the repository** and upload both datasets (`matches_full.xlsx`, `la-liga-2025-UTC.xlsx`) to your Google Colab environment.
2. **Follow the notebook cells step-by-step:**  
   - Data import and cleaning  
   - Feature extraction  
   - Initial model training (LightGBM, XGBoost, Random Forest)  
   - Prediction generation  
   - Model evaluation and accuracy reporting  

***

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- matplotlib, seaborn (for visualization)
- openpyxl (for Excel files)

***

## Project Structure

| File/Folder                | Description                                   |
|----------------------------|-----------------------------------------------|
| `matches_full.xlsx`        | Historical match records (2019-2024)          |
| `la-liga-2025-UTC.xlsx`    | 2025-26 fixture list                          |
| `predict_score_fr.ipynb`   | Main Google Colab notebook (step-by-step code)|
| `README.md`                | Project documentation                         |

***

## Getting Started

1. **Open the notebook in Google Colab.**
2. **Upload both datasets.**
3. **Run each cell** as per instructions (no user-defined functions needed).

***

## Evaluation Criteria

- Accuracy (expectation: >50%)
- Comparison of LightGBM, XGBoost, Random Forest side-by-side

***

## Authors

- [Sujoy Das]
- Dataset source: [Sujoy-004 on GitHub]

***

## License

This project is open-source and available for academic and personal use.

***

## Acknowledgments

Special thanks to the La Liga community and open-source contributors!

***

*For support or questions, feel free to open an issue or contact the project maintainer.*

***

Let me know if you want to adjust the README sections, project title, or add setup instructions for Colab!
