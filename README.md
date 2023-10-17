# deep-voice-conversion
Transform recordings of one person’s voice into the voice of another person

#### SETUP
The forked repository is `https://github.com/nii-yamagishilab/SSL-SAS`, you can refer to it for the installation process.
The directories generated are:
- adapted_from_facebookresearch
- adapted_from_speechbrain
- adapted_from_vpc
- configs
- pretrained_models_anon_xv
- scp
- scripts
- and others containing data.

#### AUDIO CONVERSION
The directories related to the conversion are:
- xv_extract: used for the speaker vector extraction
- f0_extract_avg: used for f0 extraction (and averaging for the same speaker)
- generate_xv_f0: used for audio generation, given target speaker vector and f0
- scripts_vc: useful scripts implementing conversion pipelines, in particular
    - convert.sh[^1] converts a single source-target pair of audios, generating the output in a new directory “output”
    - covert_dataset.sh[^1] converts a portion of dataset (e.g. libri_dev trials_f), mainly for testing purposes; the outputs are generated in a new directory conversion, whose contents can be used for testing (ASV)
    - other scripts are used by these two

#### AUTOMATIC SPEAKER VERIFICATION
The automatic speaker verification test is performed using the script scripts_vc/asv.sh[^1]. It is possible to perform the test both on the original audios and on the converted ones, either by adapting F0 or not.
This script requires that the speaker vectors of the speakers in the dataset have already been extracted,  for example by running scripts_vc/convert.sh, and voice conversion has already been performed on the dataset.

Examples of conversion with ASV:
Test on generated data with F0 adaptation:
```
bash scripts_vc/convert_dataset.sh libri_dev trials_f
bash scripts_vc/asv.sh libri_dev trials_f
```

Test on generated data without F0 adaptation:
```
bash scripts_vc/convert_dataset.sh vctk_test trials_m --no_f0
bash scripts_vc/asv.sh vctk_test trials_m --no_f0
```

Test on original data:
```
bash scripts_vc/convert_dataset.sh libri_dev trials_f # converts audios, too, if not already converted
bash scripts_vc/asv.sh libri_dev trials_f --original
```

#### GAN TRAINING
The directory gan-trainimg-xv contains the scritps and configuration files to train the HiFi-GAN, adding a cosine similarity term on the target and output's speaker vectors to the loss (to be tested and to adjust the terms' weigths).
scripts_vc/gan_train.sh[^1] fine-tunes the GAN to convert a specific source-target pair, maximizing the cosine similarity of output and target's speaker vectors.
A method to save or use the fine-tuned GAN is not provided, as its behavior is not tested.

[^1]: the script should be executed from the home directory (deep-voice-conversion)
