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
### Download the dataset:
1. First clone the repo, then cd into the root folder of the repo.
2. Download the datasets from this drive link: https://drive.google.com/drive/folders/1UfxlcT6akOF923KDTH3cVldejXBMM_Nb?usp=share_link
3. Unzip the downloaded datasets and place the "datasets" folder in the root directory of the project. Make sure the directory structure is as follows:\
    .\
    |---- datasets/\
    |---- |---- forest-cover-type/\
    |---- |---- churn-modelling/\
    |---- |---- eye-movements/\
    |---- |---- higgs-boson/\
    |---- |---- competitive-data-science-predict-future-sales/

### Using pyenv and virtualenv (recommended but lengthier way):
1. Install pyenv. <br />
    On linux debian based systems, install pyenv by following this guide (just install pyenv, don't install specific version of python yet): https://itslinuxfoss.com/install-use-pyenv-ubuntu/ <br /> 
   For Mac, follow this guide just install pyenv, don't install specific version of python yet): https://londonappdeveloper.com/installing-python-on-macos-using-pyenv <br />
    For Windows, the process is a bit lengthier (haven't tested). Follow this guide: https://github.com/pyenv-win/pyenv-win 
2. After installing pyenv, run:
```
pyenv install 3.10.6
```
3. After successfully installing python 3.10.6, go to the root folder of the project and run:
   ```
   pyenv local 3.10.6
   ```
4. Now install virtualenv by running:
   ```
   python3 -m pip install virtualenv
   ```
5. After installation, run:
   ```
   python3 -m venv cap5610_project
   ```
6. This creates a new directory cap5610_project/ in the root folder.
7. Activate the virtualenv by doing the following if you are on a mac or a linux based system:
   ```
   source cap5610_project/bin/activate
   ```
   If on a windows based system, run:
   ```
   env/Scripts/activate.bat //In CMD
   env/Scripts/Activate.ps1 //In Powershel
   ```
8. Now run:
   ```
   python3 -m pip install -r requirements.txt
   ```
9. After requirements are installed, run:
   ```
   python3 main.py
   ```
10. After training has finished, you can deactivate the virtualenv by running:
   ```
   deactivate
   ```

### Running without virtual env (not recommended but way easier):
1. Install dependencies using:
   ```
   python3 -m pip install numpy torch scikit-learn pandas pytorch_tabnet
   ```
2. Run the training script:
   ```
   python3 main.py
   ```
