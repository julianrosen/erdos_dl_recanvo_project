Results of experimentation with different model architectures and feature sets.

Contents:
- `ast_logistic`: AST feature set, using logistic regression classifier
- `hubert_logistic`: HuBERT feature set, using logistic regression classifier
- `hubert_dense`: HuBERT feature set, using a neural network with 2 or more dense layers as classifier
- `huber_fine_tune`: HuBERT feature extractor + 2 dense layers, unfreeze HuBERT and train
- `mel_cnn` : Mel Spectogram as feature extractor + 4 2D Convolution layers + 2 linear FC classifier
- `helpers`: some helper functions
