//maxPool2d 
//kernel_size=3 stride=2
//output one feature map per kernel
__kernel void maxpool2d(
	const int input_size,
	const int output_size,
	__global float *input_im,
    __global float *restrict output_im)
{
	int channels = get_global_id(0);//get output channel index
	
	input_im += channels * input_size * input_size;
	output_im += channels * output_size * output_size;

	//loop over output feature map
	for(int i = 0; i < output_size; i++)//row
	{
		for(int j = 0; j < output_size; j++)//col
		{
			//find the max value in 3x3 reigon 
			//to be one element in the output feature map
			float tmp = 0.0;

			#pragma unroll 1
			for(int k = 0; k < 3; k++)//row
			{
				#pragma unroll
				for(int l = 0; l < 3; l++)//col
				{
					float value = input_im[(i * 2 + k) * input_size  + j * 2 + l ];
					if(value > tmp)
						tmp = value;
				}
			}
			//store the result to output feature map
			*output_im = tmp; 
			output_im++;
		}
	}
}

//3x3 convolution layer
//output one feature map per kernel
__kernel void conv2d3x3(
	const int input_channels, const int input_size,
	const int pad, const int stride,
	const int start_channel, //start_channel is for 1x1 feature map in fire layer
	const int output_size,
	__global float* input_im,
	__global const float* filter_weight,
	__global const float* filter_bias,
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

					#pragma unroll
					for(int m = 0; m < 3; m++)
					{
						int w = j * stride + m - pad;
						if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
						{
							tmp += input_im[k * input_size * input_size + (i * stride + l - pad) * input_size + j * stride + m - pad] \
                               * filter_weight[9 * k + 3 * l + m];
						}
					}
				}
			}

			//add relu activation after conv
			*output_im = (tmp > 0.0) ? tmp : 0.0;
			output_im++;
		}
	}
}

//1x1 convolution layer
//output one feature map per kernel
__kernel void conv2d1x1(
	const int input_channels, const int input_size,
	__global float *input_im,
	__global const float* filter_weight,
	__global const float* filter_bias,
	__global float *restrict output_im)
{
	int filter_index = get_global_id(0); // 0 - (output_channels - 1)

	filter_weight += filter_index * input_channels;

	float bias = filter_bias[filter_index];
	
	output_im += filter_index * input_size * input_size;//start_channel is for 1x1 feature map in fire layer

	//loop over output feature map
	//out
	for(int i = 0; i < input_size; i++)
	{
		for(int j = 0; j < input_size; j++)
		{
			float tmp = bias;

			#pragma unroll 6
			for(int k = 0; k < input_channels; k++)
			{
				tmp += input_im[k * input_size * input_size + i * input_size + j] * filter_weight[k];
			}
			//add relu after conv
			*output_im = (tmp > 0.0) ? tmp : 0.0;
			output_im++;
		}
	}
}

//last layer use a 13 x 13 avgPool layer as classifier
//one class score per kernel
__kernel void avgpool2d(
	__global float* input_im,
	__global float *restrict output_im)
{
	int class_index = get_global_id(0);//get class score index

	input_im += 169 * class_index;
	
	float tmp = 0.0f;

	for(int i = 0; i < 169; i++)
	{
		tmp += input_im[i];
	}

	output_im[class_index] = tmp / 169.0;
}
