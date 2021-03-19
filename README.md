# Raw-Stereo-Audio-Autoencoder

This project aims to generate stereo audio after training on stereo data. The first step was to build an autoencoder in 
order to recreate various sinus. Then, a new autoencoder for stereo audio was done et tested on a new sinus dataset and finally on music dataset. 

## Code structure
### models/

It contains a .txt file with a link to a Google Drive with trained models

### Notebook

It is the notebook version of src codes. It can easily be run on colab with GPU. There are a sinus training part but also a training on piano samples.

### src/

All codes can be found in the directory 'src'. This directory is seperated in two: 'raw_audio_xxx.py' and 'sinus_xxx.py' files. As it is written in the introduction, I first worked on sinus generation so files corresponding to this step are named 'sinus_xxx.py'. Then 'raw_audio_xxx.py' files are focus on audio data. 
For each approach, we follow the same steps:
- creating/loading the dataset : 'xxx_dataset.py' (or 'dataset_generator.py')
- making autoencoder: 'xxx_autoencoder.py'
- training autoencoder on data: 'xxx_train.py'
- generating new data w/ autoencoder: 'xxx_generator.py' (or 'model_generator.py)

N.B: The VAE is not really efficient at this moment and I am not satisfy of the result, I advise you to use a classic autoencoder with variable kernel size
