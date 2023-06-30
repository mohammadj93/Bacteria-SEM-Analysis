from skimage.morphology import skeletonize, skeletonize_3d
from skimage import data
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.util import invert

import numpy as np
import math 
from numpy.random import rand
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from skimage.measure import find_contours
import cv2

from skimage.measure import profile_line

import numpy as np

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
    color_coded_length=np.zeros(img.shape,np.uint8)

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
    # LOOP OVER THE CONTOURS
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
        N=np.shape(skeleton)
        # Extract the x and y values from the Points array. notice x and y are extracted by indices 1 and 0 respectively because image rows are associated with y
        y=Points[0]
        x=Points[1]

        points = np.c_[x, y]

        if np.size(x)<10:
            continue
        # Nearest neighbor of several particle tracks should be done very easily 
        clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
        G = clf.kneighbors_graph()
        T = nx.from_scipy_sparse_array(G)
        order = list(nx.dfs_preorder_nodes(T, 0))
        xx = x[order]
        yy = y[order]
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
        xx = x[opt_order]
        yy = y[opt_order]

        xf=xx[[*range(0,xx.shape[0],10)]+[xx.shape[0]-1]] #ignore every ... element of the array for smoothing it!
        yf=yy[[*range(0,xx.shape[0],10)]+[xx.shape[0]-1]] #ignore every ... element of the array for smoothing it!

        if np.size(xf)<4:
            continue
        
        t = np.arange(len(xf))
        ti = np.linspace(0, t.max(), 30)
        
        xi = interp1d(t, xf, kind='cubic')(ti)
        yi = interp1d(t, yf, kind='cubic')(ti)

    ##    plt.imshow(draw)
    ##    plt.plot(x,y,'C1',marker = '.',linestyle = 'None')
    ##    plt.plot(xi,yi,marker = '*')
    ##    plt.gca().set_aspect('equal', adjustable='box')
        
        
        dy_dx = np.gradient(yi, xi)
        theta=np.arctan(dy_dx)
        # Calculate the perpendicular slope of the curve at each point
        perp_slope_vec = np.tan(np.pi/2+theta)
        # perp_slope_vec = -1/dy_dx
        perp_slope_vec = np.nan_to_num(perp_slope_vec, nan=0)
        perp_slope_vec[np.abs(perp_slope_vec)>100]=100
        
##        with open('data.txt', 'a') as f:
##            np.savetxt(f,perp_slope_vec, header='Array'+str(i))

        j=5;
        plotN=-1
        if i==plotN:
            plt.imshow(draw)
            plt.plot(xi,yi)
            #print(perp_slope_vec)
        SLOPE=[]
        WIDTH=[]
        for slope in perp_slope_vec[5:-5]:
            # Extract the x and y coordinates of the point
            x0, y0 = (xi[j], yi[j])
            # Calculate the y-intercept of the line
            y_intercept = y0 - slope*x0

            LL=20*(1/(1+np.exp(abs(slope)-1))+0.2);
            # Generate the x values over the specified range
            x = np.linspace(xi[j]-LL, xi[j]+LL, 10)
            # Calculate the y values of the line at each x value
            y = slope*x + y_intercept

            box=(1,1,N[1]-1,N[0])
            line=(x,y)
            RL=reduce_line_to_box(line,box)

            start=(RL[0][1],RL[0][0])
            end=(RL[1][1],RL[1][0])
            if i==plotN:
                plt.plot([start[1],end[1]],[start[0],end[0]],'r-',lw=0.5)
            v123=profile_line(draw, start, end)
    ##        print('width=',find_band_width(v123),',slope=',slope)
    ##        plt.plot(v123)

            SLOPE.append(slope)
            WIDTH.append(find_band_width(v123))
            j=j+1;
        #calculate the length
        length = length_of_curve(N[1]-yi, xi)*RES
        bact_length.append(length);

        slope_=np.array(SLOPE)
        slope_=remove_outliers(slope_)
        width_=np.array(WIDTH)*RES
        width_=remove_outliers(width_)

        std_slope=np.std(slope_)
        std_theta=np.std(np.mod(np.abs(theta),np.pi))
        mn_width=np.mean(width_)
        length=length+sum(width_)/len(width_)

        with open(fname+'_SlopeWidthLength'+'.csv', 'a') as f:
            np.savetxt(f,np.array([i,std_slope,std_theta,mn_width,length]).reshape(1, -1),\
                       delimiter=',',fmt='%.8f')
    ##    text = f"slope={std_slope:.2f},width={mn_width:.2f},length={length:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # BGR color format
        thickness = 1
    ##
    ##    # put the text on the image
    ##    AAA=cv2.putText(draw, text, (50, 50), font, font_scale, color, thickness)
    ##
    ##    # display the image
    ##    cv2.imwrite('ZZZZ'+str(i)+'.png', AAA)

    ##    plt.show()
        color_coded_length = cv2.bitwise_or(np.uint8(draw*length/4), color_coded_length)
        cv2.putText(dst, str(i), (int(xi[0]), int(yi[0])), font, font_scale, color, thickness)

    plt.imshow(dst,cmap='jet')
    plt.colorbar(plt.cm.ScalarMappable(norm=None,cmap='jet'))

    plt.savefig(fname+'_LengthColored.svg',format='svg', dpi=1200)
    plt.clf()


    cv2.destroyAllWindows()
