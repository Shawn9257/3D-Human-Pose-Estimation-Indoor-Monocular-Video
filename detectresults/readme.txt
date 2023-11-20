The image file here is the test images.
The color, depth, and infrared folders are derived from the dataset as test data (here the contents within the depth and infrared folders are only used to show differences from the inference results; the training data does not include these images);

depth2 is the model result image, where the pixel values are 0-1;
depth255 is the model result image, where the pixel values range from 0-255;
3D is the result of pose estimation and monocular depth estimation, some images appear deformed because some of the skeletal keypoints were initialized and the point was occluded. The constraints were applied when reasoning out the results, but the constraints for the occluded keypoints were not enabled when generating this result.
