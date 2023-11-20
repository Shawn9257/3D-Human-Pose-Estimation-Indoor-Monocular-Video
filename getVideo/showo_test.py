"""
    This script shows how to iterate the dataset and how to use the data stored in the json files.
    It will show color image, depth map and infrared map and project the 3D annotation into the 2D views.
 """
import json
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from TrafoUtil import *
from DrawUtils import *
import imageio
# decide which data to show
# PATH_TO_DATA = './KinectDatasets/data/training/'  # MKV train
PATH_TO_DATA = './KinectDatasets/data/validation/'  #MKV val

# PATH_TO_DATA = './data/captury_train/'  # CAP train
# PATH_TO_DATA = './data/captury_test/'  # CAP val
# PATH_TO_DATA = './data/captury_test_obj_non_frontal/'  # CAP val_ss

# load skeleton annotations
with open(PATH_TO_DATA + 'anno.json', 'r') as fi:
    anno = json.load(fi)
print('Loaded a total of %d frames.' % len(anno))

# load Kinect SDK predictions
with open(PATH_TO_DATA + 'pred_sdk.json', 'r') as fi:
    pred_sdk = json.load(fi)
print('Loaded a total of %d frames.' % len(pred_sdk))

# load camera matrices
with open(PATH_TO_DATA + 'calib.json', 'r') as fi:
    calib = json.load(fi)

# load info
with open(PATH_TO_DATA + 'info.json', 'r') as fi:
    info = json.load(fi)

# iterate frames
for fid, frame_anno in enumerate(anno):
    print('Frame %d: %d person(s)' % (fid, len(frame_anno)))
    print('Location %d' % info['location'][fid])
    print('Subject', info['subject'][fid])

    calib_frame = list()
    for cid in calib['map'][fid]:
        calib_frame.append(calib['mat'][cid])

    # iterate individual persons found
    for person_anno in frame_anno:
        coord3d = np.array(person_anno['coord3d'])  # Save already as modified
        vis = np.array(person_anno['vis'])

        # show in available cameras
        num_kinect = len(calib_frame) // 2  # because each kinect has a depth and a color frame
        for kid in range(num_kinect):
            color_cam_id, depth_cam_id = cam_id(kid, 'c'), cam_id(kid, 'd')
            img = imageio.imread(PATH_TO_DATA + 'color/' + '%d_%08d.png' % (color_cam_id, fid))
            depth = depth_uint2float(imageio.imread(PATH_TO_DATA + 'depth/' + '%d_%08d.png' % (depth_cam_id, fid)))
            infrared = imageio.imread(PATH_TO_DATA + 'infrared/' + '%d_%08d.png' % (depth_cam_id, fid))
            
        # def depth_uint2float(depth_map):
        #     upper = depth_map[:, :, 0].astype(np.float32) * 256
        #     lower = depth_map[:, :, 1].astype(np.float32)
        #     return upper + lower            
            # plt.imshow(depth)
            # print(depth.size)
            # print(depth[0])
            # print(depth[0].size)
            # print(depth[0][0])
            # print(depth[0][0].size)
            # imageio.imwrite(PATH_TO_DATA + 'depth2/' + '%d_%08d.png' % (depth_cam_id, fid),depth)
            # print("write at "+ PATH_TO_DATA + 'depth2/' + '%d_%08d.png' % (depth_cam_id, fid))


            # if len(infrared.shape) == 2:
            #     infrared = np.stack([infrared, infrared, infrared], -1)
            # infrared = infrared_scale(infrared, [30, 150])

            # # project 3d coordinates into view
            coord2d_c = project_from_world_to_view(coord3d, color_cam_id, calib_frame)
            coord2d_d = project_from_world_to_view(coord3d, depth_cam_id, calib_frame)
            world = trafo_cam2world(coord3d,depth_cam_id,calib_frame)
            print(world)
            # print(world.size)
            # print(world[0])
            # print(world[0].size)
            # print(world[0][0])
            # print(world[0][0].size)
            print(coord3d)
            # # show data
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
