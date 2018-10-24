
import numpy as np
import os
import cv2
from .colors import get_color
''' example
{  
  "objects":[  
    {  
      "prob":1.0,
      "bbox":[  
        676.0,
        240.0,
        738.0,
        301.0
      ],
      "label":"dipstick"
    },
    ...
    {  
      "prob":1.0,
      "bbox":[  
        266.0,
        184.0,
        323.0,
        238.0
      ],
      "label":"dipstick"
    }
  ],
  "file":"dataset/JPEGImages/train_image1.jpg"
}
'''

def draw_circles(data,labels=False):
    image = cv2.imread(data['file'])
    image_h, image_w, _ = image.shape
    for box in data['objects']:
        label_str = box["label"]
        xmin = box['bbox'][0]
        ymin = box['bbox'][1]
        xmax = box['bbox'][2]
        ymax = box['bbox'][3]
        prob = box['prob']
        if labels:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 5e-4 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[xmin-2,        ymin], 
                                [xmin-2,        ymin-height-13], 
                                [xmin+width+7, ymin-height-13], 
                                [xmin+width+7, ymin]], dtype='int32')  
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(xmin+7, ymin - 7), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=5e-4 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
        prob = int(box)                        
        cv2.rectangle(img=image, pt1=(xmin,ymin), pt2=(xmax,ymax), color=get_color(label), thickness=1)

    return image          