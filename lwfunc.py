from skimage.morphology import skeletonize
from skimage.io import imread
import matplotlib.pyplot as plt

import numpy as np

from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from skimage.measure import find_contours
import cv2

from skimage.measure import profile_line

import os

RES = 2 / 83 #Resolution is the actual size divided by the number of pixels equivalent to that size.

def elongation(m):
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    return (x + y**0.5) / (x - y**0.5)

def length_of_curve(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    return np.sum(ds)

def find_band_width(arr):
    first_index = None
    last_index = None
    
    for i, val in enumerate(arr):
        if val != 0:
            if first_index is None:
                first_index = i
            last_index = i
            
    if first_index is None or last_index is None:
        return 0
    
    return last_index - first_index + 1

def reduce_line_to_box(line, box):
    # line: a list of (x, y) tuples representing the line
    # box: a tuple of (xmin, ymin, xmax, ymax) representing the box
    
    x1=int(line[0][0]);x2=int(line[0][-1])
    y1=int(line[-1][0]);y2=int(line[-1][-1])
    
    if x1 < box[0]:
        y1 += (y2 - y1) * (box[0] - x1) / (x2 - x1)
        x1 = box[0]
    elif x1 > box[2]:
        y1 += (y2 - y1) * (box[2] - x1) / (x2 - x1)
        x1 = box[2]
    
    if x2 < box[0]:
        y2 += (y1 - y2) * (box[0] - x2) / (x1 - x2)
        x2 = box[0]
    elif x2 > box[2]:
        y2 += (y1 - y2) * (box[2] - x2) / (x1 - x2)
        x2 = box[2]
    
    if y1 < box[1]:
        x1 += (x2 - x1) * (box[1] - y1) / (y2 - y1)
        y1 = box[1]
    elif y1 > box[3]:
        x1 += (x2 - x1) * (box[3] - y1) / (y2 - y1)
        y1 = box[3]
    
    if y2 < box[1]:
        x2 += (x1 - x2) * (box[1] - y2) / (y1 - y2)
        y2 = box[1]
    elif y2 > box[3]:
        x2 += (x1 - x2) * (box[3] - y2) / (y1 - y2)
        y2 = box[3]
    
    return [(x1, y1), (x2, y2)]

def remove_outliers(arr):
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    lower_threshold = q25 - (1.5 * iqr)
    upper_threshold = q75 + (1.5 * iqr)
    arr_no_outliers = arr[(arr >= lower_threshold) & (arr <= upper_threshold)]
    return arr_no_outliers

def lwfunc(fname):
    print(fname + '.png')
    
    # Read SEM image which is gone through prediction process by our deep nueral network bacterial detection code based on U-net model.
    img = imread(fname + '.png')
    
    # The file name as "fname"_SlopeWidthLength.csv will contain the data of the bacteria and their standard deviation of slopes
    # which is the angles of the tangent to the main body of the bacteria; average width of the bacteria along its curve and the length of that bacteria
    if os.path.exists(fname + '_SlopeWidthLength' + '.csv'): 
        os.remove(fname + '_SlopeWidthLength' + '.csv')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # Binariz the image
    ret, bw = cv2.threshold(255-img, 160, 255, cv2.THRESH_BINARY)

    # This empty image is needed for the color-coding of the bacteria based on their length or if you change line ... based on any other desired property 
    color_coded_length = np.zeros(img.shape,np.uint8)

    # Erode the image first
    ime = cv2.erode(bw, kernel, iterations=2)
    # Blure the eroded image and binerize again:: Neccessary for removing the small nodges between the chuncks 
    bimg = cv2.blur(ime,(10,10)) 
    # Binariz the blured image again
    ret, bw = cv2.threshold(bimg, 160, 255, cv2.THRESH_BINARY)
    # Dilate the bacteria back to original size
    imd = cv2.dilate(bw, kernel, iterations=2)

    # find contours of the binarized image
    contours, heirarchy = cv2.findContours(imd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Loop over all the contours
    icounter=0;
    bact_length=[];

    for i in range(len(contours)):
        # If the bacterial size is not as of that of regular bacteria then we should rule it out!!!
        bact_size = cv2.contourArea(contours[i])
        if bact_size < 200 or bact_size > 2000:
            continue
        # For each bacgterium in the loop, create an image called "draw" that is the filled contour of that bacteriium
        draw = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.drawContours(draw, contours, i, (255,0,0), -1)

        # Binariz the draw-since the image is already binarized the cut off is put to zero
        image = draw>0
        # Perform skeletonization
        skeleton = skeletonize(draw/255)
        # Take out the nonzero elements that contain the points along the backbone of the bacterium
        Points=np.nonzero(skeleton)
        N=np.shape(skeleton) # Size of the image
        # Extract the x and y values from the Points array. notice x and y are extracted by indices 1 and 0 respectively because image rows are associated with y
        y=Points[0]
        x=Points[1]
        
        # Concatenate the x and y arrays together to run the Nearest Neighbor algorithm
        # This is to order the indices of the points so that they represent a continuous curve 
        points = np.c_[x, y]
        #get ride of the curves that are smaller than 10 points. This number can be changed based on the resolution of your curve
        if np.size(x)<10:
            continue
        # Nearest Neighbor Algorithm
        clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
        G = clf.kneighbors_graph()
        T = nx.from_scipy_sparse_array(G)
        order = list(nx.dfs_preorder_nodes(T, 0))
        x_ordered = x[order]
        y_ordered = y[order]
        paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
        mindist = np.inf
        minidx = 0
        for ic in range(len(points)):
            p = paths[ic]           # order of nodes
            ordered = points[p]    # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = ic
        opt_order = paths[minidx]
        x_ordered = x[opt_order]
        y_ordered = y[opt_order]

        # Skip every 10 elements to reduce the noise comming from the ordered points prior to smoothing the curve
        x_skipped = x_ordered[[*range(0,x_ordered.shape[0],10)]+[x_ordered.shape[0]-1]] #ignore every 10 element of the array for smoothing it!
        y_skipped = y_ordered[[*range(0,x_ordered.shape[0],10)]+[x_ordered.shape[0]-1]] #ignore every 10 element of the array for smoothing it!
        # If the smoothed curve is less than 4 elements is smaller than we could count it as a bacterium
        if np.size(x_skipped)<4:
            continue
        
        # Smoothing using cubic spline interpolation with 30 points
        t = np.arange(len(x_skipped))
        ti = np.linspace(0, t.max(), 30)
        xi = interp1d(t, x_skipped, kind='cubic')(ti) # Interpolated x
        yi = interp1d(t, y_skipped, kind='cubic')(ti) # Interpolated y
        
        # Calculate the slope of line perpendicular to the curve at each point to extract the width of the bacterium
        dy_dx = np.gradient(yi, xi)
        theta=np.arctan(dy_dx)
        perp_slope_vec = np.tan(np.pi/2+theta)
        # Remove any nan hapenning because of 90 degree or pi/2 angles which have slope equal to infinity
        perp_slope_vec = np.nan_to_num(perp_slope_vec, nan=0)
        # If the slope is higher than 100 or lower than -100 replace it with only 100. This approximation is enough for taking the width accurately
        perp_slope_vec[np.abs(perp_slope_vec)>100]=100

        SLOPE=[] # To save the slopes along the curve
        WIDTH=[] # To save the widths along the curve
        # Ignore the first 5 points in the begining of the curve to account for the effect of edges
        j=5; # j is the counter for the points along tre curve
        for slope in perp_slope_vec[5:-5]:
            # Extract the initial x and y coordinates of the point along the curve
            x0, y0 = (xi[j], yi[j])
            # Calculate the y-intercept of the line
            y_intercept = y0 - slope*x0
            # The length of the line perpendicular to the curve should not exceed the image size; LL is to insure the line does not get very large
            LL = 20*(1/(1 + np.exp(abs(slope) - 1)) + 0.2);
            # Generate the x values over the specified range
            x = np.linspace(xi[j] - LL, xi[j] + LL, 10)
            # Calculate the y values of the line at each x value
            y = slope*x + y_intercept

            box = (1, 1, N[1] - 1, N[0]) # determine the box based on the image size
            line = (x,y)
            RL = reduce_line_to_box(line, box)

            # Get the width out along the perpendicular line
            start = (RL[0][1], RL[0][0]) # starting point of the reduced line
            end = (RL[1][1], RL[1][0])   # end point      of the reduced line
            IntnsOverLine = profile_line(draw, start, end) # Intensity of the image over the perpendicular line that
                                                         # crosses the width of the bacteria to extract the width

            SLOPE.append(slope)
            WIDTH.append(find_band_width(IntnsOverLine))
            j = j + 1;
        
        # Calculate the length
        length = length_of_curve(N[1] - yi, xi) * RES
        bact_length.append(length);
        # Handling the arrays of slopes and widths
        slope_ = np.array(SLOPE)
        slope_ = remove_outliers(slope_)
        width_ = np.array(WIDTH)*RES
        width_ = remove_outliers(width_)
        # Take the standard deviation of slopes and average of the width, and the modified length for each bacterium
        std_slope = np.std(slope_)
        std_theta = np.std(np.mod(np.abs(theta),np.pi))
        mn_width = np.mean(width_)
        length = length + mn_width

        with open(fname + '_SlopeWidthLength' + '.csv', 'a') as f:
            np.savetxt(f,np.array([i,std_slope, std_theta, mn_width, length]).reshape(1, -1), \
                       delimiter = ',', fmt = '%.8f')

        # Create the color-coded image of the bacteria with their length
        color_coded_length = cv2.bitwise_or(np.uint8(draw * length / 4), color_coded_length)
        # Put the index of the bacteria in front of it just to identify them
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 1
        cv2.putText(color_coded_length, str(i), (int(xi[0]), int(yi[0])), font, font_scale, color, thickness)

    plt.imshow(color_coded_length, cmap = 'jet')
    plt.colorbar(plt.cm.ScalarMappable(norm = None, cmap = 'jet'))

    plt.savefig(fname + '_LengthColored.svg', format = 'svg', dpi = 1200)
    plt.clf()

    cv2.destroyAllWindows()
