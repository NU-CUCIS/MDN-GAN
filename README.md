# A Generative Adversarial Networks and Mixture Density Networks-Based Inverse Modeling for Microstructural Materials Design

This software is an application of a general framework combining generative adversarial networks and mixture density networks for inverse modeling in microstructural materials design. The efficacy of the proposed approach is tested on a case study where material microstructure is design given a desired optical matieral property. 

To use this software, what the algorithm requires as input is a float number from 0 to 1 representing the value for optical matrial property. Then the mixture density networks will sample various latent variable vectors, following by generative adversarial networks to produce designed microstructure images. The detailed drscription about architecture and training processing of the proposed framework can be found in the published paper given below.

## Requirements ##
* Python 3.6.3 
* Numpy 1.18.1 
* Sklearn 0.20.0 
* Keras 2.3.1 
* Pickle 4.0 
* TensorFlow 2.1.0 
* h5py 2.9.0
* Scipy 1.2.0

## Files ##
1. `model`: This folder contains scripts and saved model for mixture density networks and generative adversarial networks.
2. `data`: This folder contains two example datasets.


## How to run it
1. Run commend below, which uses trained mixture density networks to sample various latent variable vectors given a desired optical material property.
   ```
   python MDN.py
   ```
2. Run commend below, which uses sampled latent variable vectors to produce designed material microstructures.
   ```
   python ScalableG.py
   ```

## Acknowledgement
The Rigorous Couple Wave Analysis simulation is supported by Prof. Cheng Sun’s lab at Northwestern University. This work was performed under financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from NSF award CMMI-2053929, and DOE awards DE-SC0019358, DE-SC0021399.


## Related Publications ##

Y. Mao, Z. Yang, D. Jha, A. Paul, W. Liao, A. Choudhary, and A. Agrawal, “Generative Adversarial Networks and Mixture Density Networks-Based Inverse Modeling for Microstructural Materials Design,” Integrating Materials and Manufacturing Innovation, vol. 11, pp. 637–647, 2022. http://dx.doi.org/10.1007/s40192-022-00285-0 

Zijiang Yang, Dipendra Jha, Arindam Paul, Wei-keng Liao, Alok Choudhary, Ankit Agrawal. "A General Framework Combining Generative Adversarial Networks and Mixture Density Networks for Inverse Modeling in Microstructural Materials Design." Accepted by Workshop on machine learning for engineering modeling, simulation and design @ NeurIPS 2020.

## Contact
Zijiang Yang <zyz293@ece.northwestern.edu>; Ankit Agrawal <ankitag@ece.northwestern.edu>
