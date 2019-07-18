Neural Machine Translation and Linked Data
==========================================

This repository contains the source code of my bachelor thesis, Neural
Machine Translation and Linked Data. I developed the Factored
Transformer architecture in order to input BabelNet synsets alongside
words with the goal of improving state of the art NMT. Classical
linguistic features were tested as well, and actually they worked
better. The code is based on Fairseq.

Overview
--------

It is highly recommended to read Fairseq’s documentation[^1] and take a
look at its Github repository[^2] in order to have a better
understanding of our implementation. Notice that the scripts are mostly
intended for running in a high performance cluster with GPUs.

In our source code repository, the relevant directories and files are:

-   `fairseq/data/factored_language_pair_dataset.py`: Based on
    `language_pair_dataset`. In this file we can find the implementation
    for parsing and loading a dataset such that there are N source files
    (one for each factor) instead of only one. Unlike
    `language_pair_dataset`, `factored_language_pair_dataset.py` assures
    that batches are loaded such that all the source sentences refer to
    the same target sentence (ie. the features refer to the word
    sequence that has been loaded).

-   `fairseq/models`: In this directory we can find the different
    variants of the proposed architectures, as well as a required module
    for the multi-encoder architecture.

    -   `factored_composite_encoder.py`: A module for having multiple
        encoders (not specific to the Transformer) and concatenating
        their outputs. It is based on `composite_en`- `coder.py`, which
        is intended for having multiple encoders for the same source
        sequence (not factors referring to the same sequence) without
        combining their outputs.

    -   `factored_composite_encoder_sum.py`: The same as before, but
        summing instead of concatenating.

    -   `factored_transformer.py`: The implementation of the
        multi-encoder architecture with concatenation. It leverages the
        `factored_composite_encoder.py` module.

    -   `factored_transformer_sum.py`: The same as before, but summing
        instead of concatenating.

    -   `factored_transformer_one_encoder.py`: The implementation of the
        single-encoder architecture with concatenation.

    -   `factored_transformer_one_encoder_sum.py`: The same as before
        but summing instead of concatenating.

-   `fairseq/tasks/factored_translation.py`: The Fairseq task for
    executing the configuration specified in the parameters. It
    leverages `language_pair_dataset.py` in order to parse and load the
    required dataset, and it calls the specified model. It is based on
    `multilingual_translation.py`, with the key difference that instead
    of alternatively training different pairs it must use the different
    sources simultaneously, so the batches must be combined. Both the
    training step and the validation step are modified in order to
    account for this fact.

-   `fairseq/sequence_generator`: The code for loading the sequences and
    generating the translations during inference had to be modified
    similarly to `factored_translation.py`.

-   `train.py`: The generic trainer was modified to allow freezing the
    weights of a certain component at the specified epoch.

-   `preprocessing`: In this directory we can find the scripts for
    downloading, cleaning, splitting and tagging the dataset:

    -   `babelfy`: Scripts for retrieving synsets from Babelfy,
        assigning and aligning them.

    -   `iwslt14`: Scripts for getting and preparing IWSLT 14 DE-EN. The
        script `get-train-` `valid-test-iwslt14-de-en.sh` is taken from
        the authors of the baseline[^3]. In this directory there is the
        script for getting the test set from another year as well.

    -   `postagger`: Scripts for PoS tagging and aligning with the first
        tagger that we used, TreeTagger.

    -   `stanford`: Scripts for tagging and aligning classical
        linguistic features with Stanford models and Spacy.

-   `preprocess_tags.sh`: Script for Fairseq’s preprocessing
    of features. It builds the dictionaries and binarizes the data.

-   `train_tags.sh`: Script for running the training of the model. In
    this example, for the multiple-encoder architecture.

-   `generate_tags.sh`: Script for generating and evaluating
    translations with features.

-   `iwslt14_pos_synsets`: Scripts for running the experiments with PoS
    and synsets.

-   `new_iwslt14_scripts`: Scripts for running the experiments with all
    the linguistic features and lemmas.

Usage
-----

For downloading, cleaning and tokenizing the train, validation and
devtest sets:

    sbatch preprocessing/iwslt14/get-newtest-iwslt14-de-en.sh

In order to do the same but for the test set, we have to run
`get-newtest-2013.sh` the same way as before. The scripts
`run_feature_tagger_spaces.sh` and `run_align_tokensS.sh`, in
`preprocessing/stanford` `/feature_tagger_iwslt14/`, will tag the text
with classical linguistic features and align them with BPE,
respectively. `run_feature_tagger_spaces.sh` downloads the Stanford
model required for tagging as well.

On the other hand, for retrieving synsets from Babelfy:

    sbatch preprocessing/babelfy/run_get_synsets.sh

In the case that Slurm was not installed, it could be run with plain
Bash. The scripts for getting the data or tagging and aligning the
features are prepared to run without modifying any option except the
absolute path in some cases. However, in the case of the script for
retrieving scripts, the API key must be changed. By default, the script
has `KEY = ’KEY’`, which is the default one and it is allowed to do a
few calls per day.

For running the factored architectures, the Python dependencies of
Fairseq must be satisfied (see `requirements.txt`).

As an example of training and inference procedures, we will see the case
of the single-architecture with concatenation and all the classical
linguistic features, although the three required steps are analogous to
those of the other configurations. Firstly, Fairseq preprocessing must
be applied in order to build the dictionaries and binarize the data. We
have to run the corresponding preprocessing scripts. In this case,
`preprocess_stanfordS.sh` and `preprocess_tags_tokensS.sh` for words and
features, respectively, in the `new_iwslt14_scripts/` directory.

In the model code (in this case, `factored_transformer_one_encoder.py`),
we have to add the desired architecture configuration:

    @register_model_architecture('factored_transformer_one_encoder',
    'factored_transformer_one_encoder_sennrichS')
        def factored_one_encoder_iwslt_de_en(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
            args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
            args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
            args.encoder_layers = getattr(args, 'encoder_layers', 6)
            args.encoder_embed_dim_sizes = {'de_tokensS': 362, 'de_tokensS_lemmas': 115,
            'de_tokensS_pos': 10, 'de_tokensS_deps': 10, 'de_tokensS_tags': 10,
            'de_tokensS_subword_tags': 5}
            args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
            args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
            args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
            args.decoder_layers = getattr(args, 'decoder_layers', 6)
            factored_one_encoder_base_architecture(args)

For executing the training procedure:

    python train.py <WORKING_DIR> \
     --task factored_translation --arch factored_transformer_one_encoder_sennrichS \
     --optimizer adam --adam-betas '(0.9, 0.98)' \
     --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
     --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0001 --criterion \
     label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000 \
     -save-dir <MODEL_DIR> --lang-pairs de_tokensS-en,de_tokensS_lemmas-en, \
     de_tokensS_pos-en,de_tokensS_deps-en, \
     de_tokensS_tags-en,de_tokensS_subword_tags-en \
     --max-update 50000 --multiple-encoders False

For generating and evaluating translations with the trained model with
features:

    python generate.py <DESTINATION_DIR> --path <MODEL_DIR>/model.pt \
            --beam 5 --batch-size 1 --lang-pairs \ 
            de_tokensS-en,de_tokensS_lemmas-en,de_tokensS_pos-en,de_tokensS_deps-en,\
            de_tokensS_tags-en,de_tokensS_subword_tags-en \
            --task factored_translation --remove-bpe --target-lang en

[^1]: Fairseq documentation:
    <https://fairseq.readthedocs.io/en/latest/>.

[^2]: Fairseq repository: <https://github.com/pytorch/fairseq>.

[^3]: <https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh>
