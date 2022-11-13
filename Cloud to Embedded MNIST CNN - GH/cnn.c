/*
 * cnn.c
 *
 *  Created on: Nov 3, 2022
 *      Author: phili, PhiRe
 */

#include <math.h>

float * dense_layer(float * weights, float * biases, float * input, int input_size){
	//get dimensions of input
	int N = input_size; //should be 784 wide array
	int out_size = 10;
	//weights are sized: 784 x 10
	//simply do a forward array multiplication (1x784) * (784*10) = (1*10)
	
	float * result;
	result = (float *)malloc(out_size*sizeof(float));
	int i, j;
	float sum = 0.0;
	//matrix multiplication with softmax built in
	for(i=0; i < out_size; i++){
		result[i] = biases[i];
		for(j=0; j < N; j++){
			result[i] += input[j] * weights[out_size*j + i]; 
		}
		result[i] = exp(result[i]);
		sum += result[i];
	}
	//final part of softmax
	for(i=0; i < out_size; i++){
		result[i] = result[i] / sum;
	}
	return(result);
}

float * conv_layer(float*weights, float*biases, float* input, int channels, int input_shape, int filters){
	//input has shape H, W, C (height, width, channels)
	//weights have shape HH, WW, C, F (filter height, filter width, channels, filter number)
	//biases have shape F
	//stride of 1
	//pad of 1

	int C = channels;
	int H = input_shape;
	int W = input_shape;

	int F = filters; //number of filters
	int HH = 3; //kernel height
	int WW = 3; //kernel width
	
	
	//padding input
	float* padded_input;
	padded_input = (float*) malloc(sizeof(float)*(H+2) * (W+2)*C); //make sure to delete this after function finishes calculating
	int i, j, k;
	for(i = 0; i < H+2; i++){
		for(j=0; j<W+2; j++){
			for(k=0; k<C; k++){
				if(j!=0 && j!=H+1 && i!=0 && i!=W+1){
					padded_input[(H+2)*C*i + C*j + k] = input[C*H*(i-1) + C*(j-1) + k];
				}
				else {
					padded_input[(H+2)*C*i + C*j + k] = 0.0;
				}
			}
		}
	}
	
	//dimension of result: H, W, F (since we are same padding) 
	//each filter gets a bias added; not each channel (same bias for each channel)
	//addressing 1d weights by filter number and channel number:
	//weights4d[i][j][chan][filt] == weights1d[i*n_chan*n_filt + n_chan*chan + filt]
	
	float*output;
	output = (float*) malloc(sizeof(float)*F*H*W);
	
	//convolution with built in relu
	int fil, chan, x, y;
	for(x=0; x<H; x++){
		for(y=0; y<W; y++){
			for(fil = 0; fil<F; fil++){
				output[W*F*x + F*y + fil] = biases[fil];
				for(chan=0; chan< C; chan++){
					for(i=0; i <HH; i++){
						for(j=0; j <WW; j++){
							output[W*F*x + F*y + fil] += weights[F*C*WW*i + F*C*j + F*chan +fil] * padded_input[(W+2)*C*(i+x) + C*(j+y) + chan];		
						}
					}
				}
				if(output[W*F*x + F*y + fil] < 0.0){
					output[W*F*x + F*y + fil] = 0.0;
				}
			}
		}
	}
	free(padded_input);
	return(output);
}

float * maxpool_layer(float* input, int channels, int input_shape){
	//assuming 2x2 window
	//output shape should be (H/2) * (W/2) * chan
	
	int H = input_shape;
	int W = input_shape;
	int HH =(int) H/2;
	int WW =(int) W/2;
	int C = channels;
	
	float * output;
	output = (float*) malloc(sizeof(float)*(HH)*(WW)*C);
	
	int x, y, chan, i, j;
	for(x=0; x<HH; x++){
		for(y=0; y<WW; y++){
			for(chan=0; chan< C; chan++){
				output[WW*C*x + C*y + chan] = 0.0;
				for(i=0; i <2; i++){
					for(j=0; j <2; j++){
						if(output[WW*C*x + C*y + chan] < input[W*C*(2*x+i) + C*(2*y+j) + chan]){
							output[WW*C*x + C*y + chan] = input[W*C*(2*x+i) + C*(2*y+j) + chan];
						}							
					}
				}
			}
		}
	}
	return(output);
}


//code in main
//weights w1, w2, w3, b1, b2, b3 are read in using read_txt function
l1out = conv_layer(w1, b1, input_image, 1, 28, 4); 
l2out = maxpool_layer(l1out, 4, 28);
l3out = conv_layer(w2,b2, l2out, 4, 14, 4);
l4out = maxpool_layer(l3out, 4, 14);
l5out = dense_layer(w3, b3, l4out, 196);

int max =0;
int max_index = 0;
for(int i =0; i<10; i++){
	printf(“Probability of class %d: %f \n”,i, l5out[i]); 
	if(l5out[i] > max){
		max = l5out[i];
		max_index = i;
	}
}

printf(“Predicted class is %d with probability %f\n”, max_index, max);
