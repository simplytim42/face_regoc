# Get video direct from RPI for DEV ONLY

## on RPI
export DISPLAY="127.0.01:10.0"

## on Laptop
export DISPLAY="127.0.01:10.0"

## connect to RPI
ssh user@pi -X

# Using the Code
- run `python generate_dataset.py`, follow steps to add ID and Name and then use 's' to save each confirmed face and 'q' to quit
- run `python train_model.py` to then create the .yml file containing the newly trained model
- run `python recognise.py` to test the facial recognition.