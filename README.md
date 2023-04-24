# Is Deep Learning All You Need for Tabular Data?

Inspired by a paper from Intel researchers (https://arxiv.org/abs/2106.03253), this project is a recreation for a machine learning graduate course at UCF.

Link to Kaggle notebook with all datasets: https://www.kaggle.com/code/vinnyr12/is-deep-learning-all-you-need

Datasets Available via OpenML:
1. Churn Modelling
2. Eye Movements
3. Forest Cover Type
4. Rossman Store Sales
5. Higgs Boson


## Instructions to run
1. First clone the repo, then cd into the root folder of the repo.
2. Download the datasets from this drive link: https://drive.google.com/drive/folders/1UfxlcT6akOF923KDTH3cVldejXBMM_Nb?usp=share_link
3. Unzip the downloaded datasets and place the "datasets" folder in the root directory of the project. Make sure the directory structure is as follows:
    datasets/
         forest-cover-type/
         churn-modelling/
         eye-movements/
         higgs-boson/
         competitive-data-science-predict-future-sales/
4. Now install the following packages (would shift to virtualenv later) using pip:
    ```
    pip install numpy torch scikit-learn pandas pytorch_tabnet
    ```
5. From the root folder, run: 
    ```
    python3 main.py
    ```
