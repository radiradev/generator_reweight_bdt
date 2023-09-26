#!/bin/bash
#train the model
echo "Training model..."
python3 train.py || {
echo "Error running train.py. Exiting."
exit 1
}

#generate the weights
echo "Generating weights..."
python3 reweight.py || {
echo "Error running reweight.py. Exiting."
exit 1
}

#make the plots using the weights
echo "Making plots..."
python3 plot.py || {
echo "Error running plot.py. Exiting."
exit 1
}


