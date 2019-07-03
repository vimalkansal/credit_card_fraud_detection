# Credit Card Fraud Detection Using Deep Learning (PyTorch)

## Introduction

Standard Machine Learning techniques have been applied to a variety of predictive analytics problems for quite some time. Since 2012 when at Google Prof Andrew Ng made the first breakthrough with Deep Learning techniques to image recognition, the use of Deep Learning with its myriad network architectures has proliferated to numerous domains other than image processing. One such domain is anamoly detection with credit card fraud detction as one special  case.
Worldwide use of credit card is on the rise and so are the frauds associated with credit card transactions. Australians lost almost half a billion dollars in credit card fraud in a single year. Analysis by [consumer comparison website](finder.com.au) found ‘card-not-present’ fraud rose a staggering 76 per cent in the 12 months to June 30, 2018, to 1.8 million dodgy transactions. It is of utmost importance that credit card institutions be able to recognize fraudulent credit card transactions. Traditional approach to a problem like this would be to hard code the pattern recognising rules in rules engine like platforms. However with availability of huge amount of data and exponential growth in compute power, techniques like Deep Learning to learn the patterns from data offer a better alternative. 
In this blog entry, I demonstrate the use of one of the very commonly used yet very powerful neural network architecture to recognise farudulent patterns. 

Any Deep Learning endeavour requires data - lots of data !!!. If your project is a hobby or a pet project, don't get bogged down by the question "Where do I get the data from ?" Websites like [Google Dataset Search](https://toolbox.google.com/datasetsearch) and [Kaggle](kaggle.com), [UCI Machine Learning](https://archive.ics.uci.edu/ml/index.php) are your friends. Real data from different organisations (after anonymisation ) belonging to different domains and potential candidates for application of different ML/DL techniques are made available on these websites. Data publishing organisations publish the data with the intention of AI enthusists building and training ML/DL models  and publishing them which these organisations can use themselves - win-win situation for everybody. 

Data for this blog entry comes from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/home)

The dataset consists of credit card transactions in September 2013 by european cardholders. This dataset contains transactions carried over a period 2 days and out of 284807 total transactions, 492 are marked as fraudulent. As is clear from this statistics (although it is given in the description of the dataset, the dataset is highly unbalanced : positive class (fraudulent) is only 0.172 %. Although this aspect of the data distribution in this case has already been mentioned, as an AI developer, embarking on modelling a problem like this (classification), studying the data distribution is one of the key exercises to understand your data. The dataset has been anonymised and many attributes (features) have been removed. Features (attributes) have been labelled as V1, V2,..and only numerical data is included (neural nets work on numbers !). When faced with a dataset and dataset having lots of attributes/features, one of the important data preparation steps is to keep only those features which do influence the model. Deciding on which features to keep has evolved into its own area of study called 'feature engineering'. One of the techniques to find out which features are important is called PCA - Principal Component Analysis (topic for another blog !). From the description of the dataset, it is clear that PCA has already been done, so we need not worry about that ! Only features not ransformed with PCA are 'Time' and 'Amount' . 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Although given the nature of this dataset (i.e avilability of labels), it is possible to implement it as a classification problem (Supervised learning) In this blog entry, I show the use of one neural network architecture called 'Autoencoder'. 

## Autoencoders

Autoencoders are artificial neural networks capable of learning efficient representations of the input data. This learned representation is called 'coding' and this learning happens unsupervised. Autoencoders have been used for various purposes in AI. For eaxample one of the most common requirement is that of reducing the dimension of input data.  The coding that autoencoder learns has much smaller dimension than the input. This architectural aspect of autoencoders make them very suitable for dimensionality reduction. Other use cases include unsupervised pretraining of deep neural networks, randomly generating data that looks very similar to the training data, which is called a generative model. An autoencoder can be trained on pictures of faces and then this trained network will be able to generate new faces.

The way autoencoder learns is by simply copying the input to output i.e it learns an Identity function. This might sound very simple, but by constraining the network in various ways, this seemingly simple task can be made very difficult and force autoencoder to learn interesting representations. Some of the ways an autoencoder could constrained, includes limiting the size of internal representation (i.e number of input nodes much larger than nodes in hidden layer), adding noise to the input data

To understand the working of autoencoder, I draw an anlogy with the way the mind of a chess player works. An expert chess player can memorise the positions of all the pieces by just looking at the board for 5 seconds. The magic lies in the fact that he/she can do this when the pieces are lying according to some well known pattern, which he has already experienced before ! Similalry, an autoencoder looks at the input and converts them to an efficient internal representation and then spits out something that looks similar to input. There are really two components of an autoencoder : *encoder*(recognition network) which is responsible for converting the inputs to an internal representation. Encoder is followed by a *decoder* (generative network) that converts the internal representation to the outputs. 

Autoencoder  neural network is an example of 'Unsupervised Learning'. Unsupervised learning involves building model from data with no help provided by the data in terms of any labelling. Some synonyms for 'Labels' are 'Target' or 'Target Values'. A typical "Supervised Learning" will involve having dataset consisting of example data some of which has been labelled in some way and then the data is passed through the network with network outputting what it thinks what the data is ! e.g the real data (label) may be 'Cat' but the network thinks it is 'Rabbit' ! So obviously there is difference between what the real label is and what the network predicted. To fix this (or to minimise) the error, error is 'backpropagated' through the different layers apportioning the error to different nodes (obviously no single node is culprit, all nodes must have contributed to the final error). All the nodes are thus updated (actually their weights) and this cycle repeats till we are able to reach some predecided level of error, each step improving the network. Backpropagation is "the most important" technique for Deep Learning and needs a separate blog entry of its own. However, it is sufficient to know that it is a mechanism to propagate back the error with each node getting its own portion of error. In "Unsupervised Learning" there is no labelled data !!! So what does error mean in this context ? Unless there is a target value, there is no error and nothing to backpropagate and nothing to learn !! For autoencoder network we set the target values equal to input values ! 

![](Autoencoder.png)


In the above diagram we have unlabelled input data as real numbers(x<sub>1</sub>, x<sub>2</sub> ...) We set the target values equal to inputs. The autoencoder tries to learn an Identity function, so as out put Xhat that is similar to x. 

As is clear from the diagram, an autoencoder has the same architecture as an MLP (Multilayer percptron), except for the requirement that number inputs is equal to number of neuron in output layer. In the diagram, there is just one hidden layer consisting of 3 neurons (encoder) and 6 neurons in the output layer (decoder). Since autoencoder's purpose is to output a representation of input from the coding in encoder, the outputs are also called *reconstructions*. Training error (also called reconstruction loss for autoencoders) is used to penalise the model when the reconstructions are different from inputs). As is clear from this architecture, coding has leser dimension (in above digram it is 3) than input (6 in above diagram) and we are requiring decoder to generate dimension of 6 from coding of dimension 3. So this means autoencoder can not simply copy input to output, and it is forced to learn the most important features in the input and drop the others.

So, having understood that the autoencoders produce/predict the output which is almost similar to input, how can we use autoencoder for fraudulent transaction detection ? To understand this, we must note that with an autoencoder, we try to optimize the model (i.e parameters of the model) so as to minimize the reconstruction error. Now let us assume that we feed a digit 5 and at the output we receive a reconstructed digit 5 with minimum reconstruction error. Now let us try to feed another digit which is trying to impersonate 5, when we get its reconstructed version at the outputs, it will be with a large reconstruction error. This is the logic behind using autoencoder for fraud detection.


## Implementation
There are various flvours of Autoencoders that one can implement. These flavours include : Undercomplete Autoencoders, Sparse autoencoders, DAE (Denoising Autoencoders), CAE (Contractive Autoencoders), Stacked denoising autoencoders, Deep autoencoders. I will not go into the details of these architectures here. Reader is encouraged to have a look at [this very nice entry](https://medium.com/datadriveninvestor/deep-learning-different-types-of-autoencoders-41d4fa5f7570)  
I also use Pytorch for the implementation of the network in this blog entry. 


## Preprocessing
### Standardization (Feature Scaling)
It is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data…”.

This means that before you start training or predicting on your dataset, you first need to eliminate the “oddballs”. You need to remove values that aren’t centered around 0, because they might throw off the learning your algorithm is doing. 

It is a step of Data Pre Processing which is applied to independent variables or features of data. It basically helps to normalise the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm. I will not discuss all the different types of preprocessing that one might be required to carry out when embarking on AI project. In this particular instance the provided data is mostly processed. As noted in the associated Jupyter notebook, we just look at the influence of 2 columns : "Time of transaction" and "Amount" for any influence they might have on the model . We note that Time is not a factor making any contribution to the fraudulent transactions, so we drop the column. For "Amount" column we standrdize it to unit variance and zero mean. Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

We also split the original dataset into traing and test datasets using the 80:20 rule. For the training dataset, we include only those transactions which are non fraudulent . Enforcing the requirement of minimising the reconstruction loss for this training set will force the model to learn representation which it can reproduce almost same as input. Our test dataset has both kinds of transactions. So what this means is this : a model trained in this fashion when presented with a non fraudulent entry and fraudulent entry, the reconstruction loss in the former will be significantly lower than the later !

## Model
Our model is implemented as a feed forward network consisting of 4 fully connected layers with 14, 7, 7, 29 neurons. First 2 layers act as encoder and the last 2 layers act as decoder. Note last layer has 29 nodes corresponding to 29 feature in the input data item. Following daigram shows the architecture of our network, along with the activation functions chosen for each layer. 
![](credit_card_fraud_detect.png)
### Activation Functions
An A-NN at the end of the day is a function mapper i.e it maps the inputs to outputs. Our aim of training a model is to figure out the form of this function which most closely represents the nature and form of the mapping. Now these mappings could be simple linear function or a complex function. Linear equations while easy to solve, are very limited in their capacity to learn more realistic complex patterns that exist in real life data. Role of an activation function at the output of nodes from a layer is to introduce this non-linearity before this output is fed forward to the following layer. A neural network without activation function would simply be a linear regression model, incapable of learning complex patterns. So more specifically, in A-NN we do the sum of products of inputs(X) and their corresponding Weights(W) and apply a Activation function f(x) to it to get the output of that layer and feed it as an input to the next layer.

A variety of very interesting activation functions have been discovered and it continues to be an area of active research. We will not go into the details of these functions, a user is adviced to have a look at this [Wikipedia entry](https://en.wikipedia.org/wiki/Activation_function)
Choice of which activation function to use in which situation is a combination of various things : actual knowledge of mathematics behind the working of a particular AF, experience of working with similar domain and just trying out different functions, suffices to say that the choice of activation function can be deciding factor in the the successful training and performance of the model.

## Loss Function
Loss function plays the role of a penalizer i.e it penalizes the model based on how far away model is from reality. Loss function along with an optimizer helps the model to get better and better at making predictions. Several loss functions exist for different types of problems and the choice depends on various factors like the type of problem, ease of derivative calculation etc. I will not go into the details of various loss functions, a reader is advised to have a read of [this good survey of various loss functions](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23). In our implementation we are using [Mean Squred Error loss function](https://en.wikipedia.org/wiki/Mean_squared_error)

## Optimizer
So when we say we want to train a model so it understands reality as clearly as possible, what is meant ? Training means we need to keep tweaking the parameters of model till it starts behaving the way we want it to behave. These parameters (also known as the weights are the reall juice of the model. An optimiers role is to use the loss function and continously prod the model towards optimal set of weights. A range of optimizers are available and continue to be an area of active research. For a comprehensive list and behaviour of various optimisers, [please refer to this blog entry](https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3) In my implementation I chose to use Adam optimiser, choice depends on various factors - past experience, knowledge about the behaviour or pure trial !

## Training
I trained the model for 100 epochs and plotted the losses vs epochs. As is clear from the diagram losses do nicely converge.

![](training.png)

## Evaluation

Although model training losses seem to converge nicely, we still need to convince ourselves about the efficiency of model in predicting the frudulent cases. Here we do it in following ways :

### Fraudulent vs non fraudulent reconstruction error losses.
On comparing the bar diagrams of 2 cases we can see that whereas non fraudulent cases as expected show minimum reconstruction losses, quite a few fraudulent cases do show significant reconstruction error losses, which indicates that model is indeed able to detect fraudulent cases.
### Reconstruction errors - Non Fraudulent
![](non_fraudulent_loss.png)
### Reconstruction errors - Non Fraudulent
![](fraudulent_loss.png)








