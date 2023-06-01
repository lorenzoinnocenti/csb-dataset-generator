# csb-dataset-generator

Implementation of some blurring algorithms, along with an esay way to integrate them in PyTorch neural network training. The types of blur considered are:

* out-of-focus blur, implemented by convolution with a disk-shaped kernel
* camera shake blur, implemented by adapting the algorithms presented in [[1]](#1)

All the functions used to generate kernels are in [psf_generation](psf_generation). Those functions are used by the classes in [random_degradation](random_degradation), used to generate random degradations within specified parameters. 

## Blur formulation

It's possible to denote the blurring degradation process as a function of the sharp image $\mathbf{x}$

$\mathbf{y} = \Phi (\mathbf{x})$,

where $\mathbf{y}$ is the blurred image, and $\Phi$ is the blurring process. Under the assumption of spatially invariant blur, the degradation can be represented as a convolution with a Point Spread Function (PSF) kernel $\mathbf{k}$

$\mathbf{y} = \mathbf{k} \ast \mathbf{x}$ .


Multiple types of blur exist. 

### Out-of-focus blur
Out-of-focus blur happens when the camera fails to focus the scene onto the sensor. It can be mimicked using a disk-shaped kernel, which represents how light from a point source spreads through an optical system.

Here are some examples of disk kernels, generated this package:

![alt text](images/disk_kernels/1.png) &nbsp; ![alt text](images/disk_kernels/2.png) &nbsp; ![alt text](images/disk_kernels/3.png)


### Camera shake blur
Camera shake blur happens when the camera moves while acquiring an image. Under some assumptions about the camera movement, it can be modeled with a kernel that represents the trajectory of the movement during the exposure. This kernel is not as trivial to synthesize as a disk kernel, as it can take into account complex motion patterns, depending on the desired realism of the blur. 

Here are some examples of disk kernels, generated this package:

![alt text](images/csb_kernels/33.png) &nbsp; ![alt text](images/csb_kernels/57.png) &nbsp; ![alt text](images/csb_kernels/6.png)

![alt text](images/csb_kernels/73.png) &nbsp; ![alt text](images/csb_kernels/8.png) &nbsp; ![alt text](images/csb_kernels/86.png)

In this formulation, additionally to the camera shake blur degradation, we include a noise component, as proposed in [[1]](#1). To the blurred image is applied Poisson noise, caused by the statistical nature of photon detection, and Gaussian noise, caused by the amplification of the electrical signal:

$\mathbf{y} = (\mathbf{u} + \mathbf{n})/T$ ,

$\mathbf{u} \sim \mathcal{P}(\lambda(\mathbf{\mathbf{k} \ast \mathbf{x}}))$ ,
   

$\mathbf{n} \sim \mathcal{N}(0, \sigma^2)$ , 


where $\sigma$ quantifies the thermal and electrical noise of the system, and $\lambda$ the quantum efficiency of the sensor. The parameter $T$ represents the exposure time, and controls the length of the trajectory of the blurring kernel. It is the same value as the sum of the kernel pixels. The authors use this value to simulate the tradeoff between blur and noise present in cameras: when $T$ is small, the signal range is shrunk, increasing the noise effect, and reducing the blur magnitude. The multiplication by $1/T$ is an amplification factor that serves to restore the full dynamic range of the image. This is the same tradeoff between blur and noise found in cameras.

### Object motion blur
Object motion blur happens when an object moves during the exposure process. This is the most complex type of blur, as it is a spatially variant, and cannot be modeled as a convolution. We don't address it here, as other datasets like the GoPro dataset already exist.

Real-life blurred pictures can have multiple types of blur mixed together. Depending on the blurring effects taken into account, the deblurring methods can vary.

## Use

Along with the kernel generation functions, which follow the previous formulation, in the [random_degradation](random_degradation) module it's possible to find a way to randomize the degratation, so that the trained network can reverse a various array of magnitudes of degradations.

### Out-of-focus

To achieve this, the class [random_degradation.disk_blur](random_degradation/disk_blur.py) is initialized by providing a range of disk radius values and a kernel size to the constructor. The instantiated object offers a process_image function, used to degrade an image. Specifically, when an image is passed to this function, a disk radius value is randomly selected within the specified range, a PSF kernel is created, and the image is blurred using the resultant kernel. This approach can be seamlessly integrated into the training of neural networks, as we will discuss later.

### Camera shake blur

The CSB kernel generation is used in the [random_degradation.camera_shake_blur](random_degradation/camera_shake_blur.py). This class is instantiated with ranges of motion parameters and of trajectory lengths in pixels. The object exposes a process_image function that works analogously as in the disk_blur class: each time an image is passed to the function, new motion parameters and trajectory length are randomly picked, a trajectory is created, and then it is sampled to create a kernel. This kernel is used to degrade the image by convolution, and the blurred image is returned. In this class, the exposure is fixed to 1, as we use the trajectory length parameter in the trajectory class for the same function.

The full degradation algorithm from [[1]](#1), which accounts for both blur and noise, is implemented in the class [random_degradation.noisy_csb](random_degradation/noisy_csb.py). The main mechanism is the same as the previous class, but also features the application of Poisson and Gaussian noise. In addition to the parameters that controls motion, it has as input, during instantiation, a lists of values for exposure, $\lambda$, and $\sigma$. Each time the process_image function is called on an image, random values for the three parameters are extracted from the lists and used for degrading the image, in addition to the application of the blurring kernel. In this class we do not randomize the choice of the trajectory length parameter for the trajectory class, as we want to control it with the exposure value.

### Integration in network training

The integration of these functionalities in model training is presented as dataset generators for Pytorch. 
The first dataset generator, [torch_dataset_loaders.constant_kernel_dataset](torch_dataset_loaders/constant_kernel_dataset.py), is instantiated by providing a kernel created by either the csb_psf or the disk_psf, previously described. The kernel is then stored as an attribute of the object during instantiation. When a sample is required from the dataset, the corresponding sharp image is retrieved from its designated path and convolved with the stored kernel to produce the degraded image. 

The second dataset generator, the [torch_dataset_loaders.generic_degradation_dataset](torch_dataset_loaders/generic_degradation_dataset.py), is instantiated by providing a function that takes an image as input and produces another image as output. In our approach, we utilize the previously implemented process_image functions from the random degradation classes, by passing it to this dataset generator during instantiation. The function is stored as an attribute of the object. When a sample is retrieved from this generator, the corresponding sharp image is loaded from its designated path and the stored function is applied to the image, to produce the degraded version.

Examples of the generation of dataloaders, to be used in neural network training, and a few outut examples, are in [notebooks](notebooks). 

## References
<a id="1">[1]</a> 
Boracchi, Giacomo, and Alessandro Foi. "Modeling the performance of image restoration from motion blur." IEEE Transactions on Image Processing 21.8 (2012): 3502-3517.