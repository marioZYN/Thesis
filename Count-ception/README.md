# Count-ception: Counting by Fully Convolutional Redundant Counting

This paper deals with couting problem using deep CNN architecture. In particular, the author uses Inception network as the base architecture. Counting problem is very common in the field of biology. Consider this problem : "We need to count how many cells are there in a microscope image." Since the cells are hightly overlapped, it is very easy for non-expert humans to make mistakes. As the following image shows, there are many cells overlapped.

					 ![cells](https://github.com/marioZYN/Thesis/blob/master/saved/cells.png)

The idea proposed by the author is to use a fully convolutional network based on Inception architecture. Note that there is no specific requirements for which base architcture to use. 
The key ideas:

1. redundent counting using fully convolutional networks
2. construct target images using labels

The implementation details can be found in the paper. Here I modified it to our problem. First, the original paper only deals with counting one type of object. I modify it to multiple objects. Second, I design a specific handcrafted CNN architecture to contstruct the target heatmap.

The result demo can be found in this dir.
