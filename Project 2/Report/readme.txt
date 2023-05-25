Developer : Hardik Devrangadi
devrangadi.h@northeastern.edu
2/11/2023

Links/URLs:
None.

Operating System : Macbook Pro - Apple M1 Pro - MacOS: Ventura 13.1
IDE used : Xcode Version 14.2 (14C18)

Instructions:
featureVectorDB.cpp input arguments
1.Database path
2.Feature name
3.CSV File Path

imageMatch.cpp input arguments
1.Target Image path
2.Feature name
3.CSV File Path
4.Distance Metric (ssd or intersection)
5.Number of Matches

Extensions:
1. gaborTexture - 
- Feature Vector: Gabor Filters applied in different scales and orientations
- Distance Metric: Sum-of-Squared-Difference

2. gaborTextureColor - 
- Feature Vector:
 	Texture Histogram – Gabor Filters applied in different scales and orientations
 	Color Histogram – L2 Normalized Three Dimensional BGR Histogram for entire image with 8 bins for each color
- Distance Metric: Sum-of-Squared-Difference

3. multiGaborTextureColor -
- Feature Vector:
	Applied on each quadrant of a 2x2 grid (Task 3)
 	Texture Histogram – Gabor Filters applied in different scales and orientations
 	Color Histogram – L2 Normalized Three Dimensional BGR Histogram for entire image with 8 bins for each color
- Distance Metric: Sum-of-Squared-Difference

4. middleGaborTextureColor -
- Feature Vector:
	Applied to only the center section of the image
 	Texture Histogram – Gabor Filters applied in different scales and orientations
 	Color Histogram – L2 Normalized Three Dimensional BGR Histogram for entire image with 8 bins for each color
- Distance Metric: Sum-of-Squared-Difference

Command List:
baseline
histogram
multiHistogram
texture
textureColor
middleColor
middleTexture
middleTextureColor
gaborTexture
gaborTextureColor
multiGaborTextureColor
middleGaborTextureColor

Time Travel Days:
Have used 1 time travel days. 7 remaining.