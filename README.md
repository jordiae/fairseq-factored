# Fairseq baselines

The aim of this repository is preparing all the necessary to execute a baseline using Transformer and fairseq. Also it will contain all the scripts to run the models in calcula.

## Requeriments

Before using this scripts make sure that you have installed in your virtual environment subword nmt in your environment. You can install it by running:

    pip install subword-nmt

Also install all the requeriments from fairseq by running:

    pip install -r requirements.txt

And install fairseq itself (this step if you are doing modifications or running the scripts from the same folder is optional):

    python setup.py build develop

## Preprocess

In preprocess.sh you will have the script prepared to perform the following tasks:

    - Apply BPE on the dataset (train, dev and test splints). BPE codes will be shared between both languages.

    - Create the binaries and dictionaries from the data to be used during training

The following variable have to be set to the needs of your task:

**WORKING_DIR**: Directory where train, dev and test sets are located

**SRC**: Source language prefix. Files in this language should have this prefix as extension (*.SRC)

**TGT**: Target language prefix. Files in this language should have this prefix as extension (*.SRC)

**TRN_PREF**: Name of the files containing the training data without the extension. Files should have the format TRN_PREF.SRC and TRN_PREF.TGT

**VAL_PREF**: Name of the files containing the development data without the extension. Files should have the format VAL_PREF.SRC and VAL_PREF.TGT

**TES_PREF**: Name of the files containing the test data without the extension. Files should have the format TEST_PREF.SRC and TEST_PREF.TGT

**PYTHON**: Absolute path of the python binary. If the default python is already set up use only "python"

**FAIRSEQ_DIR**: Absolute path to the Fairseq installation. If the script is used from this directory "./" is enough

**DEST_DIR**: Directory where the binaries will be located. If it does not exist it will be created by the script

**N_OP**: Number of operations when applying the BPE to the data.

To run this file in calcula:

    sbatch preprocess.ph

## TRAINING

train.sh is set up to perform the training of the transformer model by setting up the following parameters:

**WORKING_DIR**: Directory where the binaries from the preprocess are stored

**CP_DIR**: Directory where checkpoints will be stored

**PYTHON**: Absolute path of the python binary. If the default python is already set up use only "python"

**FAIRSEQ_DIR**: Absolute path to the Fairseq installation. If the script is used from this directory "./" is enough

**SAVE_UPDATES**: Number of updates between checkpoints 

The file also contains all the parementers

To run this file in calcula:

    sbatch train.sh

## GENERATE

generate.sh is set up to perform inference over the test set preprocessed before. The following parameters have to be set:

**SRC**: Source language prefix

**TGT**: Target language prefix

**DEST_DIR**: Directory where the binaries from the preprocess are stored

**CP_DIR**: Directory where checkpoints are stored

**CP**: Name of the checkpoint to employ 

**PYTHON**: Absolute path of the python binary. If the default python is already set up use only "python"

**FAIRSEQ_DIR**: Absolute path to the Fairseq installation. If the script is used from this directory "./" is enough

To run this file in calcula:

    sbatch generate.sh





