# A Vocal-Cue Interpreter for Minimally Verbal Individuals
#### Deep Learning Project for the Erdős Institute, May-Summer 2024.
  Monalisa Dutta | Sarasi Jayasekara | Rahul Krishna | Alessandro Malusà | Atharva Patil | Julian Rosen
  
## What is this project about?
#### Motivation: 
Nonverbal vocalizations play an important role in communication, particularly for individuals with few spoken words. While caretakers of minimally-verbal individuals often learn to interpret nonverbal vocalizations, the vocalizations can be difficult to interpret by people unfamiliar with the individual.  

#### The dataset: 
The ReCANVo dataset consists of ~7k audio recordings of vocalizations from 8 non-verbal or minimally-verbal individuals. These individuals have limited expressive language through verbal speech and do not typically emply alternative/augmentative communication devices or signed languages. The individuals recorded here ranged in age from 6–23 years old and included diagnoses of autism spectrum disorder (ASD), cerebral palsy (CP), and genetic disorders.

The recordings were taken in a real-world setting, in a number of long sessions held at different locations (later broken into clips), and were categorized on the spot by the speaker's caregiver based on context, non-verbal cues, and familiarity with the speaker. There are several predefined categories such as self talk, frustrated, delighted, request, etc. Caregivers could also specify custom categories. 

#### Our goal: 
Training a model, per individual, that can accurately predict labels and improving upon previous work.

#### Related work:
 There have been a few publications on classifying vocalization in the ReCANVo dataset. Most relevant to our work are the two papers:
 -  ["Transfer Learning with Real-World Nonverbal Vocalizations from Minimally Speaking Individuals,"](https://www.media.mit.edu/publications/transfer-learning-with-real-world-nonverbal-vocalizations-from-minimally-speaking-individuals/) by Narain, J., Johnson, K., Quatieri, T., Picard, R., and Maes, P. 
 -  ["ReCANVo: A database of real-world communicative and affective nonverbal vocalizations,"](https://www.nature.com/articles/s41597-023-02405-7) by Johnson, K., Narain, J., Quatieri, T., Maes, P., and Picard, R. 

#### Our approach:
 
We focused on two participants with large and varied sets of observations (P01, P05). We drop all data points that correspond to labels that have fewer than 30 occurences for the participant.

For P01 and P05, we trained a number of different deep learning models. All of there models have the same 2-part structure: 
- First, we run data through a feature extractor. The main ones we focused on are pretrained [HuBERT (Hidden unit BERT)](https://arxiv.org/pdf/2106.07447) and [AST (Audio Spectrogram Transformer)](https://arxiv.org/pdf/2104.01778). 
  - For pretrained HuBERT, we use this model to feature extract by reading off the weights after a single attention-layer hidden unit pass (pretrained HuBERT has 12 such laters - we use only the most primitive).
  - For AST, we consider many possible layers at which to feature extract. 
  - We also tried other approaches, with less success, including extracting features from MEL spectrograms "by hand" using a custom convolutional net.
- Then, we train a classifier on top of the feature extractor. There are many options here, including simply:
  - (Regularized) logisitic regression;
  - a multi-layer (2 or 3) feed forward neural net;
  - XGBoost.
  
While playing with these models, we noticed something interesting: our baseline model predictions have significantly higher accuracy on vocalizations coming from sessions that were represented in the training data. We believe this due to the model picking up on background sounds from the session. 
The model's ability to train for identifying sessions hurts the model's ability to generalize. To mitigate this, we adopted several strategies including:
- Creating the train-test split in such a way that all the data points for a few entire sessions are completely contained in the test set.
- Experimenting with adding extra layers of ambient noise or removing ambient noise to confuse the potential session recognition of the model. The added noise was taken from the [DEMAND dataset](https://www.kaggle.com/datasets/chrisfilo/demand) ([Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)). The encoder-decoder transformer for noise-cancellation was implemented from [denoiser](https://github.com/facebookresearch/denoiser)[CC-BY-NC 4.0](https://github.com/facebookresearch/denoiser/blob/main/LICENSE)

This whole process of discovery can be seen in the notebook, and we encourage you to dig in!

## How do I navigate this repository?

- The `data` directory contains a lot of utility files, but notably, *not the ReCANVo files themselves.* See below.
- The `preprocessing` directory contains some inquiries into MEL spectrograms and their use for feature generation.
- The `noise_engineering` directory details our explorations into adding background noise, and the effect this has on model performance.
- The `model_validation` directory contains a number of exploratory notebooks. These look into different combinations of ('feature-extractor','classifier') pairs. Many of these notebooks also dig into regularization of our models.
  - There are subdirectories `model_validation/helpers` and `model_validation/scripts` containing some useful snippets of code. We suggest you only take a look at these as needed.

#### Getting started with the data.
The ReCANVo dataset is pubically available for download [here.](https://zenodo.org/records/5786860).
In order to run the notebooks in this repository, you should unzip the database and place the audio files (`.wav`) files in the empty directory `data/wav` within the repository.

Some of the notebooks rely on first extracting features from these audio files, then feeding the produced PyTorch tensors into our models. Running the script `model_validation/scripts/HuBERTexport.py` will export the features used by apretrained HuBERT over the whole dataset into an empty directory `data/HuBERT_features`.


####
