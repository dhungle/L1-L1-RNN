Pytorch implementation of our paper "Designing recurrent neural networks by unfolding an l1-l1 minimization algorithm", available at https://arxiv.org/abs/1902.06522. 

### What you'll need to run the code

- Pytorch
- py-yaml
- scipy
- numpy
- matplotlib
### Data
We tested our model on the [Moving MNIST dataset](http://www.cs.toronto.edu/~nitish/unsupervised_video/). Please follow the data pre-processing instruction in the paper, or download the processed data [here](https://1drv.ms/u/s!ApHn770BvhH2cJyQS9lzhfiwReA?e=fPQy59) - including video data and dictionary initialization. 
### Testing the code
```
sh run_frame_reconstruction.sh
```
Experiment settings are store in the configuration file at configs/frame_reconstruction_configs.yaml, please at least make sure that all the paths there are correct. The code was tested with Pytorch 1.0 on Ubuntu 16

We'll keep update this repo for cleaner code, more stable training, bug removal, etc. Please let us know if you find any errors or have any comments on how to improve the code/project. 

If you find this useful, please consider citing:

    @article{le2019designing,
    title={Designing recurrent neural networks by unfolding an l1-l1 minimization algorithm},
    author={Le, Hung Duy and Van Luong, Huynh and Deligiannis, Nikos},
    journal={arXiv preprint arXiv:1902.06522},
    year={2019}
    }