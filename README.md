# MLOps

California houses with MLflow elements



\#Get code:

git clone https://github.com/juviitanenAI/MLOps.git

cd MLOps



\#Create virtual environment

python -m venv env

.\\env\\Scripts\\activate



\#check virtual env is started



\# jos tarvesta asenna riippuvuudet

pip install --upgrade pip

pip install -r requirements.txt

pip install numpy pandas scikit-learn matplotlib mlflow



\# aja koodi 

python calhousemlflow.py



\#käynnistä UI (Powershell komento)

where.exe /r C:\\ mlflow.exe

<hakemisto>

\& "<hakemisto>mlflow.exe" ui --backend-store-uri file:///C:/mlruns



\#tai oma env

mlflow ui --backend-store-uri file:///C:/mlruns

