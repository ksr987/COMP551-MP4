# COMP 551 Miniproject 4: replication of a ML paper

## Libraries 

The CARE model architecture is provided by the csbdeep package, which is
available on pip `pip install csbdeep`

## Explanation of files 

The training and test scripts (1,2,3) have argparse command line flags which can
be viewed by executing with the `--help` flag.

1_training.py: Positional args are the path to the training data to load and the
name of the model to save to `./models/<model name>`. Which model (planaria,
tribolium) is trained is dependent on the training data. Hyperparameters can be
adjusted through flags.

2_planaria_test.py: Load models as specified from a text file using the
`--load-models` flag (an example of the text file is provided in
`example_load_models.txt`) to test on the planaria test dataset. Will output
figures and csv files containing the computed RMSE and SSIM scores.

3_tribolium.py: Same `--load-models` deal as 2_planaria_test.py. Separated
because a different procedure is required to load the tribolium test data than
for the planaria.

4_figures_and_tables.py: script used to compute the plots and tables in the
report. 

util.py: Local dependency of the other scripts. Contains a number of used and unused utility functions.

environment.yml: export of the conda environment used for computation. Also
contains a bunch of packages which are _not_ required for executing the code. 
