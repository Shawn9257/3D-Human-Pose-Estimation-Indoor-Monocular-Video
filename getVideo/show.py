"""
    This script shows how to iterate the dataset and how to use the data stored in the json files.
    It will show color image, depth map and infrared map and project the 3D annotation into the 2D views.
 """
from colorsys import TWO_THIRD
import json
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import imageio
from TrafoUtil import *
from DrawUtils import *

# decide which data to show
PATH_TO_DATA = './KinectDatasets/data//validation//'  # MKV train
#PATH_TO_DATA = './data/validation/'  #MKV val

# PATH_TO_DATA = './data/captury_train/'  # CAP train
# PATH_TO_DATA = './data/captury_test/'  # CAP val
# PATH_TO_DATA = './data/captury_test_obj_non_frontal/'  # CAP val_ss

# load skeleton annotations
with open(PATH_TO_DATA + 'anno.json', 'r') as fi:
    anno = json.load(fi)
print('Loaded a total of %d frames.' % len(anno))
# print(anno)

# load Kinect SDK predictions
with open(PATH_TO_DATA + 'pred_sdk.json', 'r') as fi:
    pred_sdk = json.load(fi)
print('Loaded a total of %d frames.' % len(pred_sdk))

# load camera matrices
with open(PATH_TO_DATA + 'calib.json', 'r') as fi:
    calib = json.load(fi)
# print(calib)
# load info
with open(PATH_TO_DATA + 'info.json', 'r') as fi:
    info = json.load(fi)

# print(info)
# twoD =[]

# file1 = open(PATH_TO_DATA+"2d-1.txt","w")
file2 = open(PATH_TO_DATA+"depth.txt","w")
# iterate frames
for fid, frame_anno in enumerate(anno):
    print('Frame %d: %d person(s)' % (fid, len(frame_anno)))
    print('Location %d' % info['location'][fid])
    print('Subject', info['subject'][fid])

    calib_frame = list()
    for cid in calib['map'][fid]:
        calib_frame.append(calib['mat'][cid])
        # print(cid)
    # print("calib_frame")
    # print(calib_frame)
    # print(len(calib_frame))#8
    # print(len(calib_frame[0]))#2
    # print(len(calib_frame[0][0]))#3
    # iterate individual persons found
    
    
    
    for person_anno in frame_anno:
        coord3d = np.array(person_anno['coord3d'])  # Save already as modified
        vis = np.array(person_anno['vis'])

        # show in available cameras
        num_kinect = len(calib_frame) // 2  # because each kinect has a depth and a color frame
        for kid in range(num_kinect):
            if kid!=0:
                continue
            color_cam_id, depth_cam_id = cam_id(kid, 'c'), cam_id(kid, 'd')
            print(PATH_TO_DATA + 'color/' + '%d_%08d.png' % (color_cam_id, fid))
            img = imageio.imread(PATH_TO_DATA + 'color/' + '%d_%08d.png' % (color_cam_id, fid))
            # print(img)
            depth = depth_uint2float(imageio.imread(PATH_TO_DATA + 'depth/' + '%d_%08d.png' % (depth_cam_id, fid)))
            # print("depth")
            # print(depth)
            
            infrared = imageio.imread(PATH_TO_DATA + 'infrared/' + '%d_%08d.png' % (depth_cam_id, fid))
            if len(infrared.shape) == 2:
                infrared = np.stack([infrared, infrared, infrared], -1)
            infrared = infrared_scale(infrared, [30, 150])

            # project 3d coordinates into view
            coord2d_c = project_from_world_to_view(coord3d, color_cam_id, calib_frame)
            coord2d_d = project_from_world_to_view(coord3d, depth_cam_id, calib_frame)
            # coord2d_c2= coord2d_c[0:17]
            coord2d_d2= coord2d_d[0:17]
            # np.savetxt(PATH_TO_DATA+"2d.txt",coord2d_c2,fmt='%f',delimiter=',')
            # coord2d_c2 = str(coord2d_c2.tolist())+'\n'
            
            # 存2d
            # coord2d_c2 = coord2d_c2.tolist()
            coord2d_d2 = coord2d_d2.tolist()
            for i in range(0,14):
                if i==1 :
                    continue
                # write_str1 = '%f %f\n'%(coord2d_c2[i][0],coord2d_c2[i][1])
                write_str2 = '%f %f\n'%(coord2d_d2[i][0],coord2d_d2[i][1])

                # print(write_str)
                # file1.write(write_str1)
                file2.write(write_str2)
            
            # print("coord2d_c")
            # print(coord2d_c)
            # coord2d_d2= coord2d_d[0:17]
            # print("coord2d_d")
            # print(coord2d_d2)
            # coord3d2= coord3d[0:17]
            
            # coord2d_c2= coord2d_c[0:17]
            # print("coord3d")
            # print(coord3d)
            # coord3d_world = trafo_world2cam(coord3d, depth_cam_id, calib_frame)
            # coord3d_world1 = project_into_cam(coord3d_world, color_cam_id, calib_frame)
            # print("coord3d_world1")
            # print(coord3d_world1)
            # print(calib_frame[depth_cam_id][0])#[[370.75776052924004, 0.0, 250.6039702308239], [0.0, 369.44785254653397, 210.66287806554885], [0.0, 0.0, 1.0]]
            # print(calib_frame[color_cam_id][0])#[[1068.130398118591, 0.0, 951.8851203499812], [0.0, 1065.1311607799776, 527.8419505076936], [0.0, 0.0, 1.0]]
            # file.write(coord2d_c2)
            # print("coord2d_c")
            # print(coord2d_c)
            # twoD.append(coord2d_c)
            # print("coord2d_d")
            # print(coord2d_d)
            # print(coord3d)
            
            # show data
            # with_face = 'captury' not in PATH_TO_DATA
            # fig = plt.figure()
            # ax1 = fig.add_subplot(131)
            # ax2 = fig.add_subplot(132)
            # ax3 = fig.add_subplot(133)
            # ax1.imshow(img)
            # draw_person_limbs_2d_coco(ax1, coord2d_c, vis, order='uv', with_face=with_face)
            # ax2.imshow(depth)
            # draw_person_limbs_2d_coco(ax2, coord2d_d, vis, color='g', order='uv', with_face=with_face)
            # ax3.imshow(infrared)
            # draw_person_limbs_2d_coco(ax3, coord2d_d, vis, order='uv', with_face=with_face)
            
            
            # for coords3d_sdk_pred, vis_sdk in zip(pred_sdk[fid][kid][0], pred_sdk[fid][kid][1]):
            #     coord2d_sdk_c = project_from_world_to_view(np.array(coords3d_sdk_pred), depth_cam_id, calib_frame)
            #     draw_person_limbs_2d_coco(ax2, coord2d_sdk_c, vis_sdk, color='r', order='uv', with_face=False)
                
            # plt.show()


# file1.close()
file2.close()
