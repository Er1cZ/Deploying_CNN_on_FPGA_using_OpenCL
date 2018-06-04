# A getting started tutorial on FPGA implement of CNN using OpenCL  
## 0. Prerequisites
- Basic knowledge of CNN
- Basic knowledge of OpenCL
- Familiarity in C/C++
- Familiarity in Python, PyTorch, Numpy, PyOpenCL  

We assume you alread know the prerequisites listed above. If not, we highly recommend you to go through these first:  
- [*CS231n*](http://cs231n.stanford.edu/): An excellent Stanford open course for deep learning.
- [*Hands On OpenCL*](https://handsonopencl.github.io/): An open source two-day lecture course for teaching and learning OpenCL. It will help you go through important OpenCL concepts, OpenCL kernel programming and C/C++/Python OpenCL APIs.
- [*PyTorch*](https://pytorch.org/tutorials/): PyTorch offical tutorials.

## 1. Introduction  
### 1.1 Convolutional neural network(CNN)  
CNN is one of the most popular algorithms in deep learning during recent years. It represents the state-of-art ability in several computer vision tasks, like objective detection, image classification and image segmentation. CNN has already achieved human level on image classification and even better at some specific tasks.  

<img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/CV.jpg" width="600px"/>  

*Pic from the syllabus of CS231n*
### 1.2  Why using FPGA?  
CNN is extremely computationally expensive. Recent deep CNN models require more than 20 GFLOPs per image, which CPU can’t easily process. The common solution is to accelerate the process with powerful GPU for its great capacity of parallel computing. The bottleneck of GPU accelerators is its power consumption, which can be a very crucial factor for cloud servers or embedded systems.  

Due to its parallel architecture, FPGA is also good at parallel computing, which means it's capable of traditional data parallel and task parallel computing. Even better, FPGA can achieve [*pipeline parallel*](https://www.altera.com/documentation/mwh1391807516407.html#mwh1391807498071) by generating modified circuit and data path, which outputs a result each clock cycle. Another significant benefit of FPGA is its energy consumption. FPGA can run at the same speed as a GPU but only consumes lower than 10 percent of the power.  

There are many research about using large-scale FPGA like Intel Arria 10 to completely replace GPU in PC or workstation and accelerate both back-forward pass and forward pass process of CNN.  

Since more and more CNN applications appear on embedded systems like face recognition on smart phones and object detection on drones or robots, we decide to focus on accelerating only the forward pass of CNN on embedded systems whose resources and power consumption are limited. The FPGA we have is a Cyclone V on DE10-Nano board sponsored by Terasic and Intel which we believe can be a perfect solution — using its ARM processor as traditional controller and FPGA as a low power & low latency accelerator.  

### 1.3 Why using OpenCL?
The OpenCL standard is the first open, royalty-free, unified programming model for accelerating algorithms on heterogeneous systems. OpenCL allows the use of a C-based programming language other than Verilog HDL or VHDL for rapidly developing applications on FPGA platforms. Another benefit of OpenCL is that it is a portable, open, royalty-free standard, which is a key differentiator versus proprietary programming models. And with Intel FPGA SDK for OpenCL, we can fully leverage the unique capabilities of FPGAs to deliver acceleration performance with power efficiency and low latency.  
### 1.4 Design flow  
![Design Flow](https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/DesignFlow.PNG)
## 2. Selecting CNN model  
There are several CNN models commonly used during recent years, like AlexNet, VGG, GoogleNet and ResNet. However, most models are too computationally expensive to deploy on embedded systems. Also, applications on embedded systems often require much low latency, which means deep networks can’t fit on source-limited FPGAs, like the Cyclone V FPGA on DE10-Nano board. So the first task is to find those “efficient” models.

[*Canziani et al.(2016)*](https://arxiv.org/abs/1605.07678) make a very impressive comparison among common CNN models. The computation ability of DE10-nano is around 10 GFLOPs, so it can only afford AlexNet-level models.  

<img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/CNN_FLOPs.jpg" width="450px"/>  

*Pic from Canziani's paper*  

[*Iandola et al.(2017)*](https://arxiv.org/abs/1602.07360) propose a small CNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters that are more feasible to deploy on FPGAs and other hardware with limited memory. The lateset version [*SqueezeNet v1.1*](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) requires ~0.72 GFLOPs per image, which means it achieves AlexNet-level accuracy with ~50x fewer parameters and ~3x fewer computation. It is a perfect choice to deploy on DE10-Nano.  
## 3. Designing OpenCL kernel  
The structure of SqueezeNet v1.1 is shown in the figure below. Actually, SqueezeNet v1.1 only has 4 types of layer: 3x3 convolution layer, 1x1 convolution layer, 3x3 max pool layer and 13x13 average pool layer.  

<img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/squeezenetv1.1.PNG" width="500px"/> 

### 3.1 3x3 convolution OpenCL kernel  
Every element in the output feature map of the 3x3 convolutional layer is the dot produced of a Nx3x3 matrix from the corresponding area in the input feature maps and a Nx3x3 convolution filter weight matrix, where N is the total number of the input feature maps. The output size can be calculated as ***output_size = (input_size – 3 + 2 x pad) / stride + 1***. For each convolution filter, there will be an ***output_size x output_size*** feature map.

<img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/conv.jpg" width="500px"/>

To acclerate 3x3 convolution, we design each 3x3 convolution OpenCL kernel to calculate one output feature map so that we can compute all feature maps together in parallel. The code implement is a giant loop looping over the corresponding area of input feature maps and caculate dot product. We also add RELU activation at the end of convolution to avoid using another OpenCL kernel to do this job. If this doesn't make sense, just take a look at the code below.  
```C
//3x3 convolution layer
//output one feature map per kernel
__kernel void conv2d3x3(
	const int input_channels, const int input_size,
	const int pad, const int stride,
	const int start_channel, //start_channel is for 1x1 feature map in fire layer
	const int output_size,
	__global float *restrict input_im,
	__global const float *restrict  filter_weight,
	__global const float *restrict  filter_bias,
	__global float *restrict output_im
	)
{
	int filter_index = get_global_id(0); //get output channel index

	filter_weight += filter_index * input_channels * 9;
	float bias = filter_bias[filter_index];
	output_im += (start_channel + filter_index) * output_size * output_size;
	
	//loop over output feature map
	for(int i = 0; i < output_size; i++)
	{
		for(int j = 0; j < output_size; j++)
		{
			//compute one element in the output feature map
			float tmp = bias;
			
			//compute dot product of 2 input_channels x 3 x 3 matrix
			for(int k = 0; k < input_channels; k++)
			{
				for(int l = 0; l < 3; l++)
				{
					int h = i * stride + l - pad;
					for(int m = 0; m < 3; m++)
					{
						int w = j * stride + m - pad;
						if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
						{
							tmp += input_im[k * input_size * input_size + (i * stride + l - pad) * input_size + j * stride + m - pad] * filter_weight[9 * k + 3 * l + m];
						}
					}
				}
			}

			//add relu activation after conv
			output_im[i * output_size + j] = (tmp > 0.0) ? tmp : 0.0;
		}
	}
}
```
### 3.2 1x1 convolution OpenCL kernel  
1x1 convolution OpenCL kernel is almost the same as 3x3 convolution OpenCL kernel(Due to code similarity, source code of 1x1 convolution kernel won't be listed here, so will max pool kernel and 13x13 average pool kernel. Please see [*squeezenet.cl*](https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/blob/master/src/pyopencl/squeezenet.cl) in `/src/pyopencl` folder). It just replaces the Nx3x3 corresponding area to Nx1x1. Since there is no padding and stride is 1, the size of output feature maps is the same as input feature maps size. 
### 3.3 Max pool OpenCL kernel
The goal of max pool layers is to down sample feature maps to reduce calculation. So for each input feature map, it just picks the largest activation in every 3x3 area and pass it to the output feature map. Each 3x3 max pool OpenCL kernel calculates one output feature map.
### 3.4 13x13 average pool OpenCL kernel
SqueezeNet v1.1 uses an average pool layer as a classifier. The input of this layer is a 1000 x 13 x 13 matrix. Since there are 1000 classes in the imagenet dataset, each class score can be compute as the mean of a 13 x 13 feature map. Each 13x13 average pool OpenCL kernel computes a single class score.  
## 4. Designing OpenCL host program
### 4.1 Debugging OpenCL kernel with PyOpenCL
For the sake of simplicity, the PyTorch offical pre-trained SqueezeNet v1.1 model is used here instead of actually training one. If you are quite confident with your OpenCL kernel implement, please skip this step and jump to 4.2. We highly recommend using Python OpenCL host API PyOpenCL together with PyTorch to debug kernel implement. The reasons are as follow:
- Paraemeters and intermediate results of PyTorch CNN model can be easily convert to Numpy arrays for PyOpenCL to use.
- PyOpenCL implement of host program is much more easier than C/C++ implement.
- The outputs of CNN can be checked layer by layer by comparing PyOpenCL implement with PyTorch implement so that we can debug host program logic and kernel implement at the same time.

This [*jupyter notebook*](https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/blob/master/src/pyopencl/SqueezeNet.ipynb) will help you go through this process.  

Once it is 100 percent sure that the kernel implement is correct, compile it with Intel FPGA SDK for OpenCL to see whether there are enough resources on FPGA. Use `aoc -c` command first to save time and then use full compile to generate .aocx file. Generating .aocx file could take hours. For example:  
- check syntactic error, get resource usage estimation and see kernel performance
  - `aoc -c -v squeezenet.cl -report`  
- full compile
   - `aoc -fp-relaxed squeezenet.cl -o squeezenet.aocx -board=de10_nano_sharedonly -v -report`  

More detailed imformation about `-fp-relaxed` flag can be found [*here*](https://www.altera.com/documentation/mwh1391807516407.html#mwh1391807508278).
### 4.2 Designing C/C++ OpenCL host program  
For the DE10-Nano board to run, we still need to implement host program in C/C++. The C/C++ host program is modified based on Terasic’s OpenCL vector add example in DE10-Nano OpenCL BSP and basically a translation from the PyOpenCL version described in last chapter. The translation process might be a bit tedious but shouldn’t be too hard. Parameters of SqueezeNet v1.1 is simply stored as 1d float arrays in a `.h` file. Once finished, cross compile it to an executable application with Intel Soc-EDS. 
# 5. Optimizing OpenCL kernel
By now we've got all we need to deploy CNN on FPGA. This first version of implement takes around 4.5 seconds to classify a image which should be much more faster than using ARM processer alone.

Input image:

<img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/dog.jpg" width="400px"/> 

Result:
```
SqueezeNet on FPGA start:
kernel version 1.0

conv1 takes: 176.381 ms
block1 takes: 1028.252 ms
block2 takes: 701.646 ms
block3 takes: 973.871 ms
classifier takes: 1542.772 ms
total: 4422.923 ms

predicted label: n02106662 German shepherd, German shepherd dog, German police dog, alsatian

done
```  
Resource usage:
```
+--------------------------------------------------------------------+
; Estimated Resource Usage Summary                                   ;
+----------------------------------------+---------------------------+
; Resource                               + Usage                     ;
+----------------------------------------+---------------------------+
; Logic utilization                      ;   63%                     ;
; ALUTs                                  ;   40%                     ;
; Dedicated logic registers              ;   27%                     ;
; Memory blocks                          ;   59%                     ;
; DSP blocks                             ;   46%                     ;
+----------------------------------------+---------------------------;
```
## 5.1 Unrolling loops
According to Intel:
>The OpenCL kernel performs one loop iteration of each work-item per clock cycle. With sufficient hardware resources, you can increase kernel performance by unrolling the loop, which decreases the number of iterations that the kernel executes. To unroll a loop, add a `#pragma unroll` directive to the main loop. Keep in mind loop unrolling significantly changes the structure of the compute unit that the offline compiler creates.  

Since there are still hardware resources left and most operations is done in 1x1 convolution kernel and 3x3 convolution kernel, `#pragma unroll` is used to make use of extra resources and boost kernel performance. For instance, in 3x3 convolution kernel:
```C
#pragma unroll
for(int m = 0; m < 3; m++)
{
    int w = j * stride + m - pad;
    if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
    {
      tmp += input_im[k * input_size * input_size + h * input_size + w] \
          * filter_weights[9 * k + 3 * l + m];
    }
}
```
This is a piece of code doing Nx3x3 dot product mentioned in chapter 3.1. The origin kernel computes 1 element at a time and it will be executed Nx3x3 times; With `pragma unroll`, the kernel computes 3 elements at the same time so that it only takes Nx3 iterations. More details of unrolling loops can be found [*here*](https://www.altera.com/documentation/mwh1391807516407.html#mwh1391807501939).

Best result we get is ~2.2 seconds per image:
```
SqueezeNet on FPGA start:
kernel version 1.1

conv1 takes: 177.326 ms
block1 takes: 612.210 ms
block2 takes: 409.226 ms
block3 takes: 610.608 ms
classifier takes: 356.479 ms
total: 2165.849 ms

predicted label: n02106662 German shepherd, German shepherd dog, German police dog, alsatian

done
```
And hardware resources is almost used up:
```
+--------------------------------------------------------------------+
; Estimated Resource Usage Summary                                   ;
+----------------------------------------+---------------------------+
; Resource                               + Usage                     ;
+----------------------------------------+---------------------------+
; Logic utilization                      ;   93%                     ;
; ALUTs                                  ;   60%                     ;
; Dedicated logic registers              ;   38%                     ;
; Memory blocks                          ;   97%                     ;
; DSP blocks                             ;   60%                     ;
+----------------------------------------+---------------------------;
```
## 5.2 Static memory coalescing
According to Intel:
>Static memory coalescing is an Intel® FPGA SDK for OpenCL™ Offline Compiler optimization step that attempts to reduce the number of times a kernel accesses non-private memory.The figure below shows a common case where kernel performance might benefit from static memory coalescing:
><img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/MemoryCoalescing.PNG" width="500px"/> 

With using memory coalescing in 1x1 convolution kernel, we get a slightly better performance:
```C
__kernel void conv2d1x1(
    const int input_channel, const int input_size,
    const int output_channels,    
    __global float *input_im,
    __global const float4 *filter_weights, //Static memory coalescing
    __global const float *filter_bias,
    __global float *restrict output_im)
```
Result:
```
SqueezeNet on FPGA start:
kernel version 1.2

conv1 takes: 131.246 ms
block1 takes: 492.144 ms
block2 takes: 385.413 ms
block3 takes: 511.847 ms
classifier takes: 475.307 ms
total: 1995.957 ms

predicted label: n02106662 German shepherd, German shepherd dog, German police dog, alsatian

done
```
Resource usage:
```
+--------------------------------------------------------------------+
; Estimated Resource Usage Summary                                   ;
+----------------------------------------+---------------------------+
; Resource                               + Usage                     ;
+----------------------------------------+---------------------------+
; Logic utilization                      ;   97%                     ;
; ALUTs                                  ;   64%                     ;
; Dedicated logic registers              ;   39%                     ;
; Memory blocks                          ;   96%                     ;
; DSP blocks                             ;   54%                     ;
+----------------------------------------+---------------------------;
```
More details about static memory coalescing can be found [*here*](https://www.altera.com/documentation/mwh1391807516407.html#mwh1391807503031).
# 5. Conclusion
This is a simple and relatively naive implement aiming to help those who are getting started on this topic. The best result we get is ~2 seconds per image which should be much faster than using the ARM processor alone but not fast enough to put into practial use.

There are some techniques we think might help boost the performance further:
- Using Intel channel extension. And there is an excellent open-sourse project [*PipeCNN*](https://github.com/doonny/PipeCNN) which is ~20x faster than ours.  
- Matrix multiplication instead of convolution
- Prunning CNN model
- Compressing CNN parameters
- Using fixed-point operations
