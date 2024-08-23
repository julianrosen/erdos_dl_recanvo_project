"""Some utilities for feature extraction with added noise.

This file is intended to be imported as a module in a Jupyter notebook.
"""

import pandas as pd
from pathlib import Path
import random
import torch
import torchaudio
from tqdm.notebook import tqdm

def extract_noise(
    data_shape: torch.Size,
    noise_data: pd.DataFrame,
    sample_rate: int,
):
  """Extract a layer of noise to be applied to a data point given its shape.

  :param torch.Size data_shape: The shape of the data waveform to which the noise will be applied.
  :param df.DataFrame noise_data: Metadata of the noise set.
  :param int sample_rate: Sample rate of the data file.
  """

  # Get a random noise file and load it.
  j = random.randint(0, noise_data.shape[0]-1)
  noise_wf, noise_sr = torchaudio.load(noise_data.iloc[j].Filepath)
  if noise_sr != sample_rate:
    noise_wf = torchaudio.functional.resample(noise_wf, noise_sr, sample_rate)
  noise_length = noise_wf.shape[1]

  start = random.randint(0, noise_length - data_shape[1] - 1)
  noise_wf = noise_wf[:, start:start+data_shape[1]]
  if data_shape[0]==2:
    noise_wf = torch.stack((noise_wf.flatten(), noise_wf.flatten()))

  return noise_wf

def features_with_one_noise(
    training_data: pd.DataFrame,
    noise_data: pd.DataFrame,
    feature_getter,
    sample_rate: int,
    file_seed = None,
    snippet_seed = None,
    data_loc: Path = data_loc,
):
  """Use `model` to extract features from the dataset after adding one layer of noise to every data point.

  For each data point in `training_data`, draw a noise file from `noise_data` (using `file_seed` as RNG),
  then extract a snippet from the noise file of the appropriate length so it can be overlapped with the
  data point. Resample both files (if needed) to match `sample_rate`, overlap, and then extract features
  using `model`.

  For efficiency reasons, the function iterates over the noise files rather than the data points. This is
  to avoid loading the same noise files repeatedly, since in our application the files themselves are fewer
  and much longer than the data points, and it is therefore to be expected that the same noise file will be
  drawn multiple times. This is achieved by selecting the noise files as a sample for the noise set of the
  same size as the length of the data set.

  :param pd.DataFrame training_data: Metadata of training data.
  :param pd.DataFrame noise_data: Metadata of the collection from which to draw random noise.
  :param feature_getter: The function to be used to extract the features. Should be a function that takes pytorch tensors as input (representing waveforms of the sample rate specified below).
  :param int sample_rate: sample rate expected by the feature getter.
  :param file_seed: Seed for the choice of the file from which to draw noise.
  :param snippet_seed: Seed for the choice of a snippet of sound within the noise file.
  :param Path data_loc: Location of the data files.
  :return: List of outputs of all the data files listed in `training_data` after applying one layer of noise to each.
  """

  # Initialize seed for snippet extraction.
  random.seed(snippet_seed)

  # Make a container for the extracted features. We will access this by index
  # rather than appending, so we give it the right shape right away.
  features = [_]*training_data.shape[0]

  # Draw a sample of filepaths from the noise set, of the same size as the data
  # set. 'asg' stands for 'assigned'.
  asg_noises = noise_data.Filepath.sample(
      training_data.shape[0], # One noise file per data point
      replace = True,         # Same noise file may be drawn twice
      ignore_index = True,    # New indexing independent of that in the noise set
      random_state = file_seed,
  )

  for noise_file in tqdm(asg_noises.unique()):
    # Load the noise, resample if needed.
    noise_wf, noise_sr = torchaudio.load(noise_file)
    if noise_sr != sample_rate:
      noise_wf = torchaudio.functional.resample(
          noise_wf,
          noise_sr,
          sample_rate,
      )
    noise_length = noise_wf.shape[1]

    # Iterate over all data points to which the current noise file was assigned.
    # Load, resample, extract noise snippet, overlap, extract features.
    for i in asg_noises[asg_noises==noise_file].index:
      data_path = data_loc / training_data.loc[training_data.index[i]].Filename
      data_wf, data_sr = torchaudio.load(data_path)
      data_wf = torchaudio.functional.resample(
          data_wf,
          data_sr,
          sample_rate,
      )
      data_length = data_wf.shape[1]
      start = random.randint(0, noise_length - data_length-1)
      noise_layer = noise_wf[:, start:start+data_length]
      if data_wf.shape[0] == 2:
        noise_layer = torch.stack((noise_layer.flatten(), noise_layer.flatten()))
      data_wf += noise_layer

      features[i] = feature_getter(data_wf)

  return features

def features_with_noises(
    training_data: pd.DataFrame,
    noise_data: pd.DataFrame,
    feature_getter,
    sample_rate: int,
    min_layers: int = 0,
    max_layers: int = -1,
    bernoulli: float = 0.0,
    amplification: float = 1.0,
    attenuation: float = 1.0,
    random_state = None,
    data_loc: Path = data_loc,
):
  """Use `model` to extract features from `training_data` after adding noise.

  The function adds "layers" of noise sequentially to each data point, reducing
  the noise volume exponentially with each iteration at a rate of `attenuation`.
  The overall volume of noise can be adjusted via the parameter `amplification`.

  Minimum and maximum number of layers may optionally be specified as
  `min_layers` and `max_layers`; no maximum is considered if
  `max_layers < min_layers`.
  A random number of layers may be added after the first `min_layers`. At each
  further iteration, the function "decides" with probability `poisson` to add
  at least another layer, or stops adding them otherwise.

  Noise is extracted by choosing one of the files uniformly in `noise_data`
  and then by randomly clipping a snippet of the appropriate length from that
  file.

  All draws are generated by the seed `random_state`.

  :param pd.DataFrame training_data: Metadata of training data.
  :param pd.DataFrame noise_data: Metadata of the collection from which to draw random noise.
  :param feature_getter: The function to be used to extract the features. Should be a function that takes pytorch tensors as input (representing waveforms of the sample rate specified below).
  :param int sample_rate: Sample rate expected by the feature getter.
  :param int min_layers: Minimum number of layers to be added to each file. Default is zero.
  :param int max_layers: Maximum number of layers to be added to each file. Ignored if less than `min_layers`. Default is `-1`.
  :param float bernoulli: Probability of adding each further layer after the first `min_layers` ones. Default is zero.
  :param float amplification: Volume factor applied to all noise layers. Default is `1`.
  :param float attenuation: Rate of exponential volume decay of subsequent noise layers within each data point. Default is `1`.
  :param random_state: Seed for the random number generator used for all stochastic operations.
  :param Path data_loc: Location of the data files.
  :return: List of outputs of all the data files listed in `training_data` after applying one layer of noise to each.
  """

  # Check that the parameters make sense.
  assert 0.0 <= bernoulli < 1.0, '`bernoulli` must be non-negative and strictly less than 1.0.'
  assert -1.0 <= attenuation <= 1.0, '`attenuation` should be between -1.0 and 1.0 (both included).'
  min_layers = max(0, min_layers)

  if min_layers == 1 and (max_layers == 1 or bernoulli == 0.0) and amplification==attenuation==1.0:
    # In this case we are adding exactly one layer of noise to each file,
    # and we already have a more efficient function that does that.
    return features_with_one_noise(
        training_data,
        noise_data,
        feature_getter,
        sample_rate,
        random_state,
        random_state,
        data_loc,
    )

  # Initialize random number generator.
  random.seed(random_state)

  # Initialize container for the output data.
  t_list = []

  for filename in tqdm(training_data.Filename):
    data_wf, data_sr = torchaudio.load(data_loc / filename)
    if data_sr != sample_rate:
      data_wf = torchaudio.functional.resample(data_wf, data_sr, sample_rate)

    curr_att = amplification*attenuation
    for _ in range(min_layers):
      data_wf += curr_att*extract_noise(data_wf.shape, noise_data, sample_rate)
      curr_att *= attenuation

    max_layers -= min_layers
    # If `max_layers` is negative (i.e. initially less than `min_layers`), the max-layer
    # condition in the following loop never triggers, and the number of layers is controlled
    # purely by the Bernoulli parameter.
    while max_layers != 0 and random.random() < bernoulli:
      data_wf += curr_att*extract_noise(data_wf.shape, noise_data, sample_rate)
      curr_att *= attenuation
      max_layers -= 1

    t_list.append(feature_getter(data_wf))

  return t_list
