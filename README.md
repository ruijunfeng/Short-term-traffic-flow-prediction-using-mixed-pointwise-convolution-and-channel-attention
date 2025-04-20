# Learning traffic as videos: Short-term traffic flow prediction using mixed-pointwise convolution and channel attention mechanism

## Details of the PeMSD4 and PeMSD7 datasets

| Dataset                  | PeMSD4                           | PeMSD7                        |
|--------------------------|----------------------------------|-------------------------------|
| Location                 | San Francisco Bay Area           | District 7 of California     |
| Number of sensors        | 3,796                            | 4,817                         |
| Period of time           | 1st June 2017 to 30th June 2017 | 1st June 2017 to 30th June 2017 |
| Sampling interval        | 5 minutes                        | 5 minutes                     |
| Raster size              | (42, 34)                         | (20, 36)                      |
| Number of available time points | 8,640                    | 8,640                         |

Notice: The `longitude_latitude.csv` file contains the latitude and longitude of each sensor, while the `June\traffic_flow_processed.zip` file contains the raw traffic flow data.

## Details of the CAMPConv_MC

First, the spatiotemporal traffic raster data is converted into the proposed multi-channel data structure with a shape of `(B, 3, d, I, J)`, where `B` is the batch size, `d` is the temporal depth, and `I` and `J` are the spatial dimensions. Then, the input unit learns the periodic dependencies by mapping the multi-channel data structure into a higher-dimensional output with `C` channels, resulting in a shape of `(B, C, d, I, J)`. Next, the backbone captures spatiotemporal correlations and channel inter-dependencies without downsampling the feature maps, which helps preserve more information. Finally, the output unit compresses the `C` channels into one, treating the resulting 3D feature map as a sequence of consecutive 2D feature maps with shape `(B, 1, d, I, J)`. The prediction for the next time interval is made by further compressing `d` into a single 2D feature map with shape `(B, 1, I, J)`.

## How to cite

If you find this work useful, please cite our paper:

```bibtex
@article{feng2024campconvmc,
  title = {Learning traffic as videos: Short-term traffic flow prediction using mixed-pointwise convolution and channel attention mechanism},
  journal = {Expert Systems with Applications},
  volume = {240},
  pages = {122468},
  year = {2024},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2023.122468},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417423029706},
  author = {Ruijun Feng and Mingzhou Chen and Yaqi Song},
  keywords = {Traffic flow prediction, 3D convolution, Convolutional neural network, Channel attention mechanism, Pointwise convolution, Multi-channel data structure},
  abstract = {In the construction of intelligent transportation systems, short-term traffic flow prediction is of great significance for the advancement of traffic network management. But due to the presence of many complex factors in both spatial and temporal domains, it remains a complex and challenging task. Existing literature usually employs the convolutional neural network (CNN)-based methods in capturing spatiotemporal correlations. These CNN-based methods often use a single-channel data structure to represent different periodic patterns, which makes the model susceptible to over-parameterization when capturing periodic dependencies and prone to information loss after convolution. To overcome these limitations, this paper presents a hybrid deep learning method for short-term traffic flow prediction. In this method, a video-shaped multi-channel data structure is designed to represent different periodic patterns more efficiently. Next, a new mixed-pointwise convolution is introduced for capturing periodic dependencies without the negative impacts mentioned above. Lastly, an improved channel attention mechanism is proposed to learn channel inter-dependencies with controllable parameter usage. The proposed method is lightweight, yet highly effective. Compared to the state-of-the-art baseline method, it reduces the root mean squared error by up to 6.7% on the PeMSD4 dataset and 13.3% on the PeMSD7 dataset, while also achieving substantial improvement in two additional metrics, exhibiting strong robustness and great scalability across various settings.}
}

