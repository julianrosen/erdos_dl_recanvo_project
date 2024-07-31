This directory contains some research notebooks.

The notebooks assume existence of a directory `../data/wav`, which contains `.wav` files from the ReCANVo dataset.

 - `hubert_example.ipynb`: Using featues extracted from a pretrained HuBERT model as inputs to logistic regression.
 - `hubert_example_p05.ipynb`: Similar to above, but using participant `P05` instead of `P01`. Also computes some additional metrics, and looks at different sample weightings.
 - `hubert_unfreezing.ipynb`: Build a neural network from the bottom layers of pretrained HuBERT, with linear and softmax layers on top. Try some training with bottom layers frozen, then more training with unfrozen. Performance matches the logistic regression, but is not better.
 - `subsample_test.ipyn`: Modeling using subsampling of data to balance class sizes, as was done in the transfer learning paper. This gives results that are directly comparable to the results in that paper.
