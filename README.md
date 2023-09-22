# Short-term-traffic-flow-prediction-using-mixed-pointwise-convolution-and-channel-attention
## Details of the PeMSD4 and PeMSD7 datasets
| Dataset                  | PeMSD4                           | PeMSD7                        |
|--------------------------|----------------------------------|-------------------------------|
| Location                 | San Francisco Bay Area           | District 7 of California     |
| Number of sensors        | 3,796                            | 4,817                         |
| Period of time           | 1st June 2017 to 30th June 2017 | 1st June 2017 to 30th June 2017 |
| Sampling interval        | 5 minutes                        | 5 minutes                     |
| Raster size              | (42, 34)                         | (20, 36)                      |
| Number of available time points | 8,640                    | 8,640                         |

Notice: The longitude_latitude.csv file contains the latitude and longitude of each sensor, while the June folder contains the raw traffic flow data.
## Details of the CAMPConv_MC
First, the spatiotemporal traffic raster data is converted into the proposed multi-channel data structure with a shape of (B, 3, d, I, J), where B represents the batch size. Then, the input unit learns the periodic dependencies by mapping the multi-channel data structure into bigger output channels (B, C, d, I, J). Next, the backbone captures spatiotemporal correlations and channel inter-dependencies without downsampling the feature maps, which helps preserve more information. Finally, the output unit compresses the output channels into one and treats the 3D output feature map as a sequence of consecutive 2D feature maps (B, 1, d, I, J), the prediction of the next time interval is made by compressing d into a single 2D feature map (B, 1, I, J).
