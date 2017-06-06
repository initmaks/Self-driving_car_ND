#**Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="misc/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

This project detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

The Project
---

**Step 1:** Getting setup with Python

To do this project, you will need Python 3 along with the numpy, matplotlib, and OpenCV libraries, as well as Jupyter Notebook installed. 

Installing the Anaconda Python 3 distribution from Continuum Analytics is recommended because it comes prepackaged with many of the Python dependencies you will need for this project, makes it easy to install OpenCV, and includes Jupyter Notebook. 

Choose the appropriate Python 3 Anaconda install package for your operating system <A HREF="https://www.continuum.io/downloads" target="_blank">here</A>.   Download and install the package.

If you already have Anaconda for Python 2 installed, you can create a separate environment for Python 3 and all the appropriate dependencies with the following command:

`>  conda create --name=yourNewEnvironment python=3 anaconda`

`>  source activate yourNewEnvironment`

**Step 2:** Installing OpenCV

Once you have Anaconda installed, first double check you are in your Python 3 environment:

`>python`    
`Python 3.5.2 |Anaconda 4.1.1 (x86_64)| (default, Jul  2 2016, 17:52:12)`  
`[GCC 4.2.1 Compatible Apple LLVM 4.2 (clang-425.0.28)] on darwin`  
`Type "help", "copyright", "credits" or "license" for more information.`  
`>>>`   
(Ctrl-d to exit Python)

run the following commands at the terminal prompt to get OpenCV:

`> pip install pillow`  
`> conda install -c menpo opencv3=3.1.0`

then to test if OpenCV is installed correctly:

`> python`  
`>>> import cv2`  
`>>>`  (i.e. did not get an ImportError)

(Ctrl-d to exit Python)

**Step 3:** Installing moviepy  

We recommend the "moviepy" package for processing video in this project (though you're welcome to use other packages if you prefer).  

To install moviepy run:

`>pip install moviepy`  

and check that the install worked:

`>python`  
`>>>import moviepy`  
`>>>`  (i.e. did not get an ImportError)

(Ctrl-d to exit Python)

**Step 4:** Opening the code in a Jupyter Notebook

You will complete this project in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, run the following command at the terminal prompt (be sure you're in your Python 3 environment!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook. 
  
Description
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

[//]: # (Image References)

[image1]: ./misc/original.png "Original"
[image2]: ./misc/grey.png "Grey"
[image3]: ./misc/blurred.png "Blurred"
[image4]: ./misc/canny.png "Canny"
[image5]: ./misc/verts_orig.png "Vertices Original"
[image6]: ./misc/verts_canny.png "Vertices Canny"
[image7]: ./misc/red_lines.png "Lines"
[image8]: ./misc/green_lines.png "Green Lines"
[image9]: ./misc/raw_xy.png "X & Y Coordinates"
[image10]: ./misc/X_Y_pred.png "X & Y Predicted"
[image11]: ./misc/pred_lines.png "X & Y Predicted Lines"
[image12]: ./misc/asphalt.jpg "Asphalt Symbol"
[image13]: ./misc/reflection.png "Reflection"
[image14]: ./misc/mess.jpg "Mess"

--- 

#1. Pipeline description.


Two different pipilens were created during this project.
One of the uses the Hought Transform and another one adds KNN + Polynomial fitting for extracting the lane lines.

1.1. Hough Transform
---

Let's open the example image and see what it look like.

![alt text][image1]

Next, the image is converted to greyscale and Gaussian blur is applied 

![alt text][image2]
![alt text][image3]

After that the edges on the image are extracted using the Canny Edge detector. (with threshold set as 60 and 180 for low and high values respectively)

![alt text][image4]

As the next step only the region that can contain the road lines is extracted

![alt text][image5]
![alt text][image6]

Searching for lines using Hough Transform and drawing them

![alt text][image7]

As the result we get the pipeline that can extract the lines that from the selected region of the image.
There are could be many onf them and the image may look quite noisy. Pipeline is used to process the video **solidWhiteRight.mp4**
And the resultant video can be viewed [here](processed_videos/white.mp4)

1.1.2 Extrapolationg the lines
---

Target is to obtain single solid line that represents the road line from the image.
From multiple lines we can extract
  
Pipeline developed in the **1.1.2** produces the list of lines. 
Lines from that list are split into two arrays, left and right, based on the slope of the line calculated as following *(y2-y1)/(x2-x1)*
If the slope is less than zero, it's is added to list of left lines, and to the right otherwise.

List contains only the starting points and the ending point of the lines. Combining all of the ending and starting points and getting they average respectively, give the average points that can be used to draw a average line.
![alt text][image8]

And the resultant video can be viewed [here](processed_videos/yellow.mp4)

1.2 KNN + Polynomial fitting
---

Another idea to try was to apply KNN to split the lines in to two groups and then fit the polynomial of the first/second degree to extract the line.

Process is the samy up to the point of extracting the Canny Edges from selected region

![alt text][image6]

Then all the points that are white here, converted to the list of x and y coordinates:

![alt text][image9]

Using unsupervised clustering algorithm [Kmeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) from the [sklearn](http://scikit-learn.org/) library, with the setup of of 10 iterations and number of clusters present in data as 2, two sets of the points was obtained.

![alt text][image10]

Using [numpy](http://www.numpy.org/)'s [polyfit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) function two lines were fit to 2 clusters.

![alt text][image11]

While the processing of the video last 5 video frames prediction results are being used to make the line smother. (5 is just a random number of choice, the real life choice have to be made based on the frame frame and other details)
And the resultant video can be viewed [here](processed_videos/yellow_knn.mp4)

1.3 Hough Transform + KNN + Polynomial fitting
---

And as the last pipeline first two pipelines were combined together as they may produce different result on different scenes and averaging their output is beneficial.
Same as before last 5 video frames were used for smoothing.
And the resultant video can be viewed [here](processed_videos/yellow_knn_hough.mp4)

###2. Identify potential shortcomings with your current pipeline


One potential shortcoming is apperance of the road symbol in the middle of the road on the asphalt. Like following:

![alt text][image12]

Pipeline may recognize this dense number of white points as the line and prediction will be shifted.

Another shortcoming could be the reflection of the light from the Sun, like on the following photo:

![alt text][image13]

Reflection could be recognized as the line of it will be brighter that the line it self.

And of course:

![alt text][image14]

###3. Suggest possible improvements to pipeline

Parameters of the Canny and Hough transform have to be tweaked more. And would be good to make them adjust automatically based on the scene.

Another potential improvement would be to try to use one or two specific colours, to see if it helps to see the lines better in different weather and light conditions.
 
Would be nice to make pipeline have some sort of memory, as the road doesn't change too much every frame. If nothing happens to the car that will lead to shift of the image, information from the previous frames could be used to extract the pattern of the lines on the road.
