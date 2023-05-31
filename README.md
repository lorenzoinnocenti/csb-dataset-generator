# csb-dataset-generator

Implementation of some blurring algorithms, and a esay way to integrate them in PyTorch neural network training. The types of blur considered are:

* out-of-focus blur, implemented by convolution with a disk-shaped kernel
* camera shake blur, implemented by adapting the algorithms presented in [[1]](#1)

All the functions used to generate kernels are in [package.psf_generation](package/psf_generation). Those functions are used by the classes in [package.random_degradation](package/random_degradation). Those classes are initialized with the degradation parameters, and allow the generation of images with a random degradation each time, within the parameters set during the initialization. The degradation implemented are:

* random out of focus blur, by picking a range of radiuses for the disk kernel;
* random camera shake blur, by generating a new kernel for each image;
* random moisy camera shake blur, same as the previous but including the application of noise as in [[1]](#1);
* constant trajectory degradation, same as the previous but the trajectory is kept the same, to better show the tradeoff between noise and blur.

The random degradation classes can be integrated in the training of PyTorch neural networks as shown in [notebooks](notebooks)



## References
<a id="1">[1]</a> 
Boracchi, Giacomo, and Alessandro Foi. "Modeling the performance of image restoration from motion blur." IEEE Transactions on Image Processing 21.8 (2012): 3502-3517.