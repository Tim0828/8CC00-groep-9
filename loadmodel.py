# load the model
from keras.models import load_model

model_name = 'my_model.keras'
model = load_model(model_name)

# load the data
from loaddata import load_data

df = load_data('untested_molecules.csv')