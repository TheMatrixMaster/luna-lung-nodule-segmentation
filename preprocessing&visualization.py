from __future__ import division, print_function
import SimpleITK as sitk
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


data_dir = "D:/LUNA_CTscans/data/"
output_path = "D:/LUNA_CTscans/processed_data/"
file_list = []

for subset in os.listdir(data_dir):
	subset_dir = os.path.join(data_dir, subset)
	for mhd_file in os.listdir(subset_dir):
		if mhd_file.endswith(".mhd"):
			img_path = os.path.join(subset_dir, mhd_file)
			file_list.append(img_path)


def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 

    plt.show()


def make_mask(center, diam, z, width, height, spacing, origin):
    '''
center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
width X height : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width]) 
    # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5]) 
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix = matrix - m_min
    return(np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16))

#The locations of nodules in annotations csv
df_node = pd.read_csv('D:/LUNA_CTscans/CSVFILES/annotations.csv')
df_node['file'] = df_node['seriesuid'].apply(get_filename)
df_node = df_node.dropna()

print(df_node.head())

'''
# Looping over the image files
for fcount, img_file in enumerate(tqdm(file_list)):
	print("Getting mask for image file %s" % img_file.replace(data_dir,""))
	mini_df = df_node[df_node["file"] == img_file] 								# get all nodules associate with file
	if mini_df.shape[0] > 0:       												# some files may not have a nodule--skipping those 
		biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]			# just using the biggest node
		node_x = mini_df["coordX"].values[biggest_node]
		node_y = mini_df["coordY"].values[biggest_node]
		node_z = mini_df["coordZ"].values[biggest_node]
		diam = mini_df["diameter_mm"].values[biggest_node]

		itk_img = sitk.ReadImage(img_file)										
		img_array = sitk.GetArrayFromImage(itk_img)								# indexes are z,y,x (notice the ordering)
		num_z, height, width = img_array.shape        							# height X width constitute the transverse plane
		center = np.array([node_x, node_y, node_z])								# nodule center
		origin = np.array(itk_img.GetOrigin())      							# x,y,z  Origin in world coordinates (mm)
		spacing = np.array(itk_img.GetSpacing())    							# spacing of voxels in world coor. (mm)
		v_center = np.rint((center - origin) / spacing)  						# nodule center in voxel space (still x,y,z ordering)

		#plot_ct_scan(img_array)													# plot scan slices

		imgs = np.ndarray([3, height, width], dtype=np.float32)
		masks = np.ndarray([3, height, width], dtype=np.uint8)

		i = 0
		for i_z in range(int(v_center[2]) - 1, int(v_center[2]) + 2):
			mask = make_mask(center, diam, i_z * spacing[2] + origin[2], width, height, spacing, origin)
			masks[i] = mask
			imgs[i] = matrix2int16(img_array[i_z])
			i += 1
		np.save(os.path.join(output_path, "images_%d.npy" % (fcount)), imgs)
		np.save(os.path.join(output_path, "masks_%d.npy" % (fcount)), masks)
'''

imgs = np.load(os.path.join(output_path, 'images_780.npy'))
masks = np.load(os.path.join(output_path, 'masks_780.npy'))
for i in range(len(imgs)):
	print("image %d" % i)
	fig, ax = plt.subplots(2, 2, figsize=[8, 8])
	ax[0, 0].imshow(imgs[i], cmap='gray')
	ax[0, 1].imshow(masks[i], cmap='gray')
	ax[1, 0].imshow(imgs[i] * masks[i], cmap='gray')
	plt.show()
	raw_input("hit enter to continue: ")