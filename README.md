# wasserstein_autoencoders
Implementation of Wasserstein Autoencoders

From the paper - "Wasserstein Auto-Encoders": https://arxiv.org/abs/1711.01558

## Loss functions
### Discriminator 
The adversarial game is played between latent space terms. The 'real' z is the standard zero mean unit variance gaussian. The 'fake' is the latent code produced by the WAE. We concoct terms accordingly. 

### Generator 
There are two terms, one coming from reconstruction, and the other is the adversarial regularization term (which must now be matched adversarially to a zero mean unit variance gaussian). 


### Architecture 
Generator: Mostly borrowed from DCGAN. 
Discriminator: Since we are working with latent space terms, we use full connections for z variables (no need for convolutions as this is not an image). 

Of interest is the balancing parameter lambda to adjust regularizer importance. 

We should definitely add resnet layers to improve the capacity of the generator (not yet done). 

