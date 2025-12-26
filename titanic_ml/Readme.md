To run the project
python -m venv .env
.emv\Scripts\activate.bat
pip install -r requirements.txt
python train_model.py
python train_xgboost.py
python -m uvicorn app:app --reload


Step 0
Create a virtual environment
python -m venv ml-venv

Activate the created virtual environment
ml-venv\Scripts\activate.bat

Step 1
Install the requirements first
pip install -r requirements.txt

Step 2 
The file train_model.py is the script which trains the model, same as the task was done on the keggel notebook
Command to execute 
py train_model.py


once the model has been trained, update requirements.txt file 
pandas
scikit-learn
joblib
fastapi
uvicorn
pydantic


re-install all the dependencies
pip install -r requirements.txt


Add the required controllers in the app.py


to run the app, use below command
python -m uvicorn app:app --reload


Added 
Name: REST Client
Id: humao.rest-client
Description: REST Client for Visual Studio Code
Version: 0.25.1
Publisher: Huachao Mao
VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=humao.rest-client


In order to test apis, I created request.http
and if the extension is available, there will be a link about each request with the test SendRequest

To start the project, execute 
python -m uvicorn app:app --reload


Added XGBoost as version 2 (V2)
add xgboost in the requirements.txt
Then
pip install -r requirements.txt
