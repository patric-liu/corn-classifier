import xmltodict #pip3 install xmltodict
import os
import numpy as np
import cv2
import glob
'''
Takes the annotations from a particular video and extracts the images,
and saves them into folders corresponding to their labels. 
'''

class GetAnnotations(object):
    def __init__(self,object,padding):
        self.video_name = object
        # folder and directory management
        self.current_dir = os.path.dirname(__file__)
        self.read_path = self.current_dir + '/annotations/' + self.video_name
        self.writepath = self.current_dir + \
            '/annotation_snapshots/' + self.video_name + '/'
        if not os.path.exists(self.read_path):
            os.makedirs(self.read_path)
        # containers
        self.image_name_dict = {} # counts n of existing images for labels to prevent replacement
        # parameters
        self.padding = padding

    def get_info(self,i=0):
        ''' extract point and label information for a specific annotation 
        from an xml file'''
        label = self.doc['annotation']['objects']['object'][i]['label']
        points = self.doc['annotation']['objects']['object'][i]['points']
        x_pts = list(map(lambda x: int(x), points['x']))
        y_pts = list(map(lambda y: int(y), points['y']))
        return label, x_pts, y_pts

    def extract_images(self):
        ''' loops through each annotation file and loops through each individual
        annotation, cropping and saving the relevant parts of each frame
        '''
        j = 1
        # loop through each annotation file
        for filepath in glob.glob(os.path.join(self.read_path + '/annotations', '*xml')):
            print('proccessing labeled frame number', j, '\n')
            j += 1

            # prepare the relevant image for sampling
            timestamp = self.get_timestamp(filepath)
            image_path = self.read_path + '/images/img_' + str(timestamp) + '.png'
            self.img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)

            # read the xml file to a dictionary
            with open(filepath) as fd:
                self.doc = xmltodict.parse(fd.read())
           
            i = 0
            # loop through each annotation within the file
            while True:
                try:
                    label, x_pts, y_pts = self.get_info(i)

                    # naming and directory management
                    try:
                        self.image_name_dict[label] += 1
                    except KeyError:
                        self.image_name_dict[label] = 0              
                        newpath = self.writepath + label
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                    # get bounding boxes and save files
                    xmin, xmax, ymin, ymax = self.get_coordinates(x_pts,y_pts)
                    self.crop_save(filepath, xmin, xmax, ymin, ymax, label)
                    i += 1
                except IndexError:
                    break
            
    def crop_save(self, filepath, xmin, xmax, ymin, ymax, label):
        ''' makes a cropped copy of the current frame and saves it'''
        index = str(self.image_name_dict[label])
        image_save_path = self.writepath + '/' + label + '/' + index + '.png'
        cropped_img = self.img[ymin:ymax,xmin:xmax]
        cv2.imwrite(image_save_path, cropped_img)
  
    def get_coordinates(self,x_pts, y_pts):
        ''' take two lists containing the set of points that make up an annotation
        and return a bounding box for the shape'''
        image_x_size = 640  # TEMPORARY
        image_y_size = 480  # TEMPORARY
        xmin = min(x_pts) - self.padding
        xmax = max(x_pts) + self.padding
        ymin = min(y_pts) - self.padding
        ymax = max(y_pts) + self.padding
        return xmin,xmax,ymin,ymax

    def get_timestamp(self,filepath):
        '''grabs timestamp from filename'''
        timestamp = ''
        for l in filepath:
            try:
                if timestamp[-5:] == 'anno_':
                    timestamp = ''
            except IndexError:
                pass
            timestamp += l
        timestamp = timestamp[:-4]
        return timestamp

if __name__ == "__main__":
    video_name = 'video1' 
    action = GetAnnotations(object = video_name,
                            padding = 0 # number of boundary pixels
                            )
    action.extract_images()
