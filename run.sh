#!/bin/bash

#check if an argument was passed to the script
if [ -z "$1" ]
then
echo "No flux specified. Please specify either 'dune' or 'flat'."
exit 1
fi

#set the config file based on the argument passed
if [ "$1" == "dune" ]
then
CONFIG_FILE="config/dune.py"
elif [ "$1" == "flat" ]
then
CONFIG_FILE="config/flat.py"
else
echo "Invalid flux specified. Please specify either 'dune' or 'flat'."
exit 1
fi

echo "Using config file: $CONFIG_FILE"
python3 $CONFIG_FILE || {
echo "Error running $CONFIG_FILE. Exiting."
exit 1
}


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


