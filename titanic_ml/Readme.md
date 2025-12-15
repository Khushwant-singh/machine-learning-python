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
