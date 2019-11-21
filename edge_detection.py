import os
import sys
import argparse
import cv2
import numpy as np
import pickle

import dehaze.dark_prior_channel as dpc
import edge_detector.canny_edge_detector as ced


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def convert_to_bw(folder):
    for filename in os.listdir(folder):
        im_gray = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite( 'gray_image'+filename, im_gray)

def split_to_value(folder):
    for filename in os.listdir(folder):
        im_color = cv2.imread(os.path.join(folder,filename), 1)
        im_hsv = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(im_hsv)
        cv2.imwrite( 'value_image'+filename, v)

def dehaze_bw_constrast(infolder, outfolder):
    dehazer = dpc.DarkPriorChannelDehaze(
                wsize=15, radius=80, omega=1.0,
                t_min=0.25, refine=True)
    for filename in os.listdir(infolder):
        if not filename.endswith('.jpg'):
            continue
        # if the file was processed, ignore it
        if os.path.exists(os.path.join(outfolder, filename)):
            continue
        print("filename: {}".format(filename))
        # read color image
        im_color = cv2.imread(os.path.join(infolder,filename),1) 
        print("original shape: {}".format(im_color.shape))
        
        # rescale the image
        scale_percent = 70
        width = int(im_color.shape[1] * scale_percent / 100)
        height = int(im_color.shape[0] * scale_percent / 100)
        dim = (width, height)
        im_color = cv2.resize(im_color, dim, interpolation=cv2.INTER_AREA)
        print("resized shape: {}".format(im_color.shape))

        # dehaze the image
        im_dehaze = dehazer(im_color)
        #cv2.imwrite( 'dehaze_image'+filename, im_dehaze)

        # change contrast
        im_dehaze = apply_brightness_contrast(im_dehaze, contrast=64) 

        # convert to value
        #im_hsv = cv2.cvtColor(im_dehaze, cv2.COLOR_BGR2HSV)
        #h,s,v = cv2.split(im_hsv)
        #cv2.imwrite( 'value_image'+filename, v)

        # store as b&w
        im_gray = cv2.cvtColor(im_dehaze, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite( 'gray_image'+filename, im_gray)

        # canny edge 
        imgs = [im_gray]
        detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, 
                    lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
        imgs_final = detector.detect()
        im_edge = imgs_final[0].astype(np.uint8)
        cv2.imwrite(os.path.join(outfolder, filename), im_edge)
         
        # draw contour
        im_contour = im_edge.copy()
        contours, hierarch = cv2.findContours(im_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("number of contours: {}".format(len(contours)))
        #import pdb;pdb.set_trace()
        cnts = []
        for contour in contours:
            maxx = 0
            minx = 1000000000
            maxy = 0
            miny = 1000000000
            for i in range(len(contour)):
                x, y = contour[i][0]
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y 
                if y > maxy: maxy = y
            if maxx-minx >= minwidth and maxy-miny >= minheight: 
                #print("x range: [{},{}], y range: [{}, {}]".format(minx, maxx, miny, maxy))
                cnts.append(contour)
        print("number of fillterd contour: {}".format(len(cnts)))
        canvas = np.zeros_like(im_edge)
        cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(outfolder, "contour_"+filename), canvas)

        # save contour to pickle
        pickle.dump( cnts, open(os.path.join(pickled_folder, filename[:-4]+'.p'), "wb" ) )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="infolder", 
                help="folder contains original images")
    parser.add_argument("-o", "--output", dest="outfolder", 
                help="folder contains edgy images")
    parser.add_argument("-p", "--pickle", dest="pickle", 
                help="folder contains pickled contours")
    parser.add_argument("--minwidth", dest="minwidth", type=int,default=50,
                help="the minimum width of contour")
    parser.add_argument("--minheight", dest="minheight", type=int, default=50,
                help="the minimum height of contour")

    args = parser.parse_args()
    if args.infolder is None or args.outfolder is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.infolder):
        print("folder {} does not exists".format(args.infolder))
        sys.exit(1)

    if not os.path.exists(args.outfolder):
        print("folder {} does not exists".format(args.outfolder))
        sys.exit(1)

    minwidth = args.minwidth
    minheight = args.minheight

    if args.pickle is None:
        pickled_folder = args.outfolder
    else:
        pickled_folder = args.pickle
    dehaze_bw_constrast(args.infolder, args.outfolder)

