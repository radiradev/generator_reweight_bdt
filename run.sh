#!/bin/bash
#train the model
config_name=$1

echo "Training model..."
python3 train.py $config_name || {
echo "Error running train.py. Exiting."
exit 1
}

#generate the weights
echo "Generating weights..."
python3 reweight.py $config_name || {
echo "Error running reweight.py. Exiting."
exit 1
}

#make the plots using the weights
echo "Making plots..."
python3 plot.py $config_name || {
echo "Error running plot.py. Exiting."
exit 1
}


