Multiple Instance Learning (MIL), a type of Weakly Supervised model, is used to detect whether a medical claim is fraudulent or not.

MIL observes collections of claims, grouped by providers which are labeled as fraudulent / not fraudulent, in order to identify the fraudulent claims and their characteristics. 

2 different MIL models are analyzed.

1) The first model implemented is an instance-space algorithm, made of four fully-connected layers. The number of outputs of each fully-connected layers are 256, 128, 64 and 1. The last layer is a MIL pooling layer, which takes the last outputted instance probabilities as input and outputs the bag probability.

2) The second model is a multiple instance neural network implementing an embedded-space MIL algorithm. The difference from the first proposed MIL model is that this embedded-space network directly learns a bag representation rather than aggregating learned instance representations. This should produce better bag classification accuracy.
This model uses residual connections. If the first model pooled instances into a bag representation in a final step, this model can be described as using a kind of residual pooling, where each fully-connected layer contributes a pooled bag representation to a final sum bag representation. The initial FC layer produces a bag-feature vector, and each FC layer learns weights contributing to the
final sum of residuals. In this way, a bag representation is learned. The size of all fully-connected layers is 128. ReLU and MIL pooling are used at every step and, at the end of the network, the final bag representation is used to output the bag label via a final dense layer with one neuron and Sigmoid activation.

[Link to data.zip.](https://drive.google.com/file/d/1kAx6yy2LIHBZosQVsWFQYpS5ekaDJUj4/view?usp=sharing)
