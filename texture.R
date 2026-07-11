library(terra)
before_image <- rast("before_aoi_clip.tif")
after_image <- rast("after_aoi_clip.tif")
plotRGB(x = before_image, r = 1, g = 2, b = 3)
plotRGB(x = after_image, r = 1, g = 2, b = 3)


# # Morphology
# 
# # create ball kernel
# ball <- matrix(c(NA,NA,1,NA,NA,
#                  NA,1,1,1,NA,
#                  1,1,1,1,1,
#                  NA,1,1,1,NA,
#                  NA,NA,1,NA,NA),nrow = 5,byrow = T)
# ball
# 
# # create result
# dilated <- focal(x = input_image[[1]], w = ball, fun = max)
# closed  <- focal(x = dilated, w = ball, fun = min)
# 
# plot(closed, col = grey(1:255/255))




# # Convolution
# #kernel <- matrix(c(-1,-1,-1,0,0,0,1,1,1),nrow = 3,byrow = T)
# kernel <- matrix(rep(1,25),nrow = 5,byrow = T)
# kernel
# sma <- focal(x = input_image[[1]], w = kernel,fun=median)
# plot(x=sma, col=grey(1:255/255))

## Sobel Operator

#specify filter kernels for X- and Y-direction
KernelX <- matrix(c(-1,0,1,-2,0,2,-1,0,1), nrow = 3, byrow = T) #  insert the missing coefficients to define Kernel X as in slide 16
KernelY <- matrix(c(1,2,1,0,0,0,-1,-2,-1), nrow = 3, byrow = T) # insert the missing coefficients to define Kernel Y as in slide 17

#show kernels
KernelX
KernelY


# complete the lines below to apply both kernels separately using focal() (summing up the element-wise products instead of mean)
#store results (raster layers) in variables SobelX and SobelY
before_SobelX <- focal(x = before_image[[1]], w = KernelX, fun = sum)
before_SobelY <- focal(x = before_image[[1]], w = KernelY, fun = sum)

#compute final result as magnitude of results of both Kernels
before_Sobel <- sqrt(before_SobelX^2+before_SobelY^2)

#visualize result
plot(x=before_Sobel, col=grey(1:255/255))


# complete the lines below to apply both kernels separately using focal() (summing up the element-wise products instead of mean)
#store results (raster layers) in variables SobelX and SobelY
after_SobelX <- focal(x = after_image[[1]], w = KernelX, fun = sum)
after_SobelY <- focal(x = after_image[[1]], w = KernelY, fun = sum)

#compute final result as magnitude of results of both Kernels
after_Sobel <- sqrt(after_SobelX^2+after_SobelY^2)

#visualize result
plot(x=after_Sobel, col=grey(1:255/255))


# einfache Change Analysis
result_change <- abs(after_Sobel - before_Sobel)
plot(result_change, col = grey(1:255/255), main = "Change analysis")

result_thresh <- global(result_change, "mean", na.rm = TRUE)[1,1] +
  global(result_change, "sd", na.rm = TRUE)[1,1]

result_change_mask <- result_change > result_thresh
plot(result_change_mask, main = "Change mask")