# Fully Convolutional Networks for Semantic Segmentation

This paper describes how we can convert a normal CNN to F-CNN which is known as fully Convolutional Network. The ideas are the following:  
1. Fully connected layers in normal CNN can only receives fixed dimension of inputs. We can view the fully connected layers as convolutional layers.

2. If the network only contains convolutional layers, then it can receive any size of inputs. If a normal CNN is trained on w \* h \* d images, and we have 10 classes for output. After softmax layer, we get a 1 \* 10 output. Each value is the probabilty for the input image being that particular class. After transfering the CNN into F-CNN, we can input images with larger size. The output will be w2 \* h2 \* 10, where w2 and h2 depend on the network architecture(padding, striding, pooling). This output is not a single probability but a heat map. Each pixel value in the output is determined by its **receptive field**.

3. After getting the heat map, we can use upsampling techniques to map the heap map size back to the original image size. In this paper, the authors propose a technique called **transpose convolution**, which uses a filter larger than the input size to enlarge the input. This filter is learnable. If we have ground-truth semantic segmentation images, we can construct an end-to-end, pixel-level F-CNN for semantic segmentation.

A demo can be found in this repo.
