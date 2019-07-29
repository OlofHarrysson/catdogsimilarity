# bigger resnet model
# sgd with momentum
# only the negative cosine. Different margins. Seems like data points are inherently close in cosine space after coming through the network anyways
# Cyclic LR tricks
# Regularize network
# Add small branch for classification
# Dig deeper into how to viz data to understand better. What about the outliers?
# Sometimes dogs doesn't really look like other dogs. We force the network to cluster all dogs together with all other dogs. Can we relax this a bit so that we don't just do one cluster? 
# pos margin=1, neg margin=0
# LOok at data points that are incorrectly classified


# ~~~~~~~~~~~ REFERENCE IMAGES ~~~~~~~~~~~
# Curate the reference images to the vals. Should work good for top1. Find out which images we are having a bad time with, add reference images from train that correlate to those troublemakers / remove.
# Find all the wrong predictions. Remove the reference img that makes it a bad prediction from future reference ims. Repeat.

# Compare one reference images at a time against all val images. If the reference image is closer to the wrong class it's a bad reference image.

# Two types of outliers. Outlier in ref image, outlier in pred image. How does each one affect result and how to deal with each one?