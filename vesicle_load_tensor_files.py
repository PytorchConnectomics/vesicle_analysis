import edt
import h5py
import numpy as np
#from xvfbwrapper import Xvfb
#from pyvirtualdisplay import Display
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
import os
import napari

scaling_factors = np.array([30, 8, 8])

def load_data(neuron_file_path, vesicle_file_path):
    print("begin loading data")

    #print("current dir:", os.getcwd())
    #print("expected path", os.path.join(os.getcwd(), 'vol0_mask.h5'))
    
    #with h5py.File("vol0_mask.h5", 'r') as f:

    #INSTEAD USE MASK
    with h5py.File(neuron_file_path, 'r') as f:
        #hard code the file path bc of issues opening h5 files
        neuron_data = f['main'][:]

    print("successfully loaded first file")
    print("shape of {neuron_file_path}: ", neuron_data.shape)

    #with h5py.File("vol0_vesicle_ins.h5", 'r') as f:
    with h5py.File(vesicle_file_path, 'r') as f:
        #hard code the file path bc of issues opening h5 files
        vesicle_data = f['main'][:]

    print("successfully loaded second file")
    print("shape of {vesicle_file_path}: ", vesicle_data.shape)

    return neuron_data, vesicle_data

    #load the neuron mask?? & align


def calculate_distance_transform(neuron_data):
    print(np.unique(neuron_data))
    #return_edt = edt.edt(1 - neuron_data.astype(np.uint32), anisotropy=(8, 8, 30), black_border=True, order='F')
    return_edt = edt.edt(1 - neuron_data, anisotropy=(8, 8, 30))
    return return_edt

def identify_vesicles_within_perimeter(labeled_vesicles, distance_transform, perimeter_distance_threshold):
    vesicles_within_perimeter = []
    vesicles_within_perimeter_labels = set()

    for region in regionprops(labeled_vesicles):
        coords = np.clip(region.coords, 0, np.array(distance_transform.shape) - 1)

        if np.any(distance_transform[coords[:, 0], coords[:, 1], coords[:, 2]] <= perimeter_distance_threshold):
            vesicles_within_perimeter.append(region.coords)
            vesicles_within_perimeter_labels.add(region.label)
    return vesicles_within_perimeter, vesicles_within_perimeter_labels


def find_all_vesicle_counts(neuron_file_path, labeled_vesicles, distance_transform, distance_threshold_voxels):

    with h5py.File(neuron_file_path, 'r') as f:
        neuron_labels = f['main'][:]

    vesicles_within_perimeter= dict()
    
    for neuron_label in np.unique(neuron_labels):
        if neuron_label == 0:
            continue  # Skip background

        #mask for the current neuron segmentation region
        neuron_mask = (neuron_labels == neuron_label)

        
        one_neuron = np.where(neuron_labels == neuron_label, neuron_labels, 0)

        #find coordinates of the perimeter for the current neuron region
        #distance_transform_for_neuron = distance_transform * neuron_mask
        distance_transform_for_neuron = calculate_distance_transform(one_neuron)
        
        print("neuron_mask shape:", neuron_mask.shape)
        mask_shape = neuron_mask.shape

        #create a mask with the same shape as the original neuron_data
        
        expanded_mask = np.copy(neuron_labels)
        
        #add a perimeter
        perimeter_mask = distance_transform_for_neuron <= distance_threshold_voxels
        
        # Add the perimeter to the original mask
        expanded_mask[perimeter_mask] = 1

        print(expanded_mask)

        #export the mask including the perimeters
        with h5py.File(f"{neuron_label}_perimeter_mask.h5", 'w') as f:
            f.create_dataset('main', data=expanded_mask, compression='gzip')
            print(f"successfully saved perimeter mask for {neuron_label}")


        #need to fix this part - use convert_coords
        vesicles_coords = np.column_stack(np.nonzero(labeled_vesicles)) #need to scale to the new res?
        vesicles_within_current_neuron = []

        for coord in vesicles_coords:
            #add checker - might need to fix offset issues..
            print("coord: ", coord)
            #(8,8,30), (128,128,120) -> (30,8,8), (120, 128,128)
            converted_coord = convert_coords(coord, (188, 4096, 4096), (8, 8, 30), (47, 256, 256), (128, 128, 120)) #change into neuron file coords
            print("converted_coord: ", converted_coord)
            if (0 <= converted_coord[0] < mask_shape[0] and 0 <= converted_coord[1] < mask_shape[1] and 0 <= converted_coord[2] < mask_shape[2]):
                if neuron_mask[tuple(converted_coord)] and perimeter_mask[tuple(converted_coord)]:
                    vesicles_within_current_neuron.append(converted_coord)
            else:
                print("alignment error")

        #store vesicles within the current neuron region
        vesicles_within_perimeter[neuron_label] = vesicles_within_current_neuron
        print("num of vesicles within current neuron:, ", len(vesicles_within_current_neuron))

    print(vesicles_within_perimeter)
    return vesicles_within_perimeter



#needs to be fixed - not outputting valid coords
def convert_coords(original_coords, original_shape, original_res, target_shape, target_res):
    scale_factors = np.array(target_res) / np.array(original_res)
    print("scale_factors: ", scale_factors)
    
    target_coords = []

    scaled_coord = np.array(original_coords) * scale_factors
        
    #convert type
    target_coord = np.round(scaled_coord).astype(int)
        
    #check if coords valid
    if all(0 <= target_coord[i] < target_shape[i] for i in range(3)):
        target_coords.append(tuple(target_coord))
    else:
        print(f"coord {original_coords} out of bounds after scaling: {target_coord}")
    
    return target_coord



def create_mesh(positions):
    scaled_positions = positions.astype(np.float32) * scaling_factors
    return pv.PolyData(scaled_positions)

def visualize_data(neuron_mesh, vesicles_within_perimeter, vesicles_within_perimeter_labels, labeled_vesicles, perimeter_positions):
    #vdisplay = Xvfb()
    #vdisplay.start()
    #pv.start_xvfb()

    #display = Display(visible=0, size=(800, 600))
    #display.start()

    plotter = pv.Plotter()
    plotter.add_mesh(neuron_mesh, color='blue', opacity=0.3, point_size=5, render_points_as_spheres=True)

    for vesicle in vesicles_within_perimeter:
        if len(vesicle) > 0:
            vesicle_mesh = create_mesh(vesicle)
            plotter.add_mesh(vesicle_mesh, color='green', point_size=10, render_points_as_spheres=True)

    for region in regionprops(labeled_vesicles):
        if region.label not in vesicles_within_perimeter_labels:
            vesicle_mesh = create_mesh(region.coords)
            plotter.add_mesh(vesicle_mesh, color='red', point_size=10, render_points_as_spheres=True)

    perimeter_mesh = create_mesh(perimeter_positions)
    plotter.add_mesh(perimeter_mesh, color='yellow', opacity=0.1, point_size=5, render_points_as_spheres=True)

    plotter.add_axes()
    plotter.show_grid()
    # Set the grid spacing to 10 microns
    plotter.set_scale(zscale=0.001)
    plotter.set_scale(xscale=0.001)
    plotter.set_scale(yscale=0.001)
    plotter.show_grid(xtitle='X axis (10 microns)', ytitle='Y axis (10 microns)', ztitle='Z axis (10 microns)')

    plotter.show()


#also - neuron_predictions
def load_calculate_and_visualize_neuron_and_vesicles(neuron_file_path, vesicle_file_path, perimeter_distance_threshold_nm=1000):

    neuron_data, vesicle_data = load_data(neuron_file_path, vesicle_file_path)
    
    #need to check this?
    labeled_vesicles, num_vesicles = label(vesicle_data, return_num=True)
    #also need to label neurons?

    print("labeling done")

    print("start dist transform")
    distance_transform = calculate_distance_transform(neuron_data)
    print("end dist transform")

    '''
    perimeter_distance_threshold_nm = 1000
    # Convert the perimeter distance threshold from nanometers to voxel distances
    perimeter_distance_threshold_voxels = (
        perimeter_distance_threshold_nm / scaling_factors[0],  # x
        perimeter_distance_threshold_nm / scaling_factors[1],  # y
        perimeter_distance_threshold_nm / scaling_factors[2]   # z
    )
    '''

    #needs to be three coords...? fix this
    perimeter_distance_threshold_voxels = 10 #100

    '''
    neuron_coords = np.column_stack(np.nonzero(neuron_data))
    vesicles_within_perimeter, vesicles_within_perimeter_labels = identify_vesicles_within_perimeter(
        labeled_vesicles, distance_transform, perimeter_distance_threshold_voxels)
    '''

    #visualize_heatmap(vesicles_within_perimeter_labels)

    print(distance_transform)

    #create a mask with the same shape as the original neuron_data
    expanded_mask = np.copy(neuron_data)
    
    #add a perimeter
    perimeter_mask = distance_transform <= perimeter_distance_threshold_voxels
    
    #add the perimeter to the original mask
    expanded_mask[perimeter_mask] = 1

    print(expanded_mask)

    #export the mask including the perimeters
    with h5py.File("perimeter_mask.h5", 'w') as f:
        f.create_dataset('main', data=expanded_mask, compression='gzip')
        print("successfully saved perimeter mask to perimeter_mask.h5")

    perimeter_positions = np.argwhere(perimeter_mask)

    
    neuron_positions = np.column_stack(np.nonzero(neuron_data))
    neuron_mesh = create_mesh(neuron_positions)


    #convert perimeter distance threshold from nm to voxel distances
    distance_threshold_nm = 1000
    distance_threshold_voxels = (
        distance_threshold_nm / scaling_factors[0], #z
        distance_threshold_nm / scaling_factors[1], #y
        distance_threshold_nm / scaling_factors[2] #x
    )

    #find vesicles within the perimeter for each neuron segmentation
    vesicles_within_perimeter = find_all_vesicle_counts(neuron_file_path, labeled_vesicles, distance_transform, 10)

    for neuron_label, vesicles in vesicles_within_perimeter.items():
        print(f"Neuron label {neuron_label}:")
        print(f"Number of vesicles within perimeter: {len(vesicles)}")
        print(f"Vesicles coordinates: {vesicles}")



    '''
    print("shape of labeled_vesicles: ", labeled_vesicles.shape)
    density_map = calculate_density_map(labeled_vesicles)
    visualize_heatmap(density_map, neuron_data)
    '''


'''
#given a 3d neuron prediction h5 file w/ segmentations
#use the mask file - "7-13_pred_filtered.h5"
def process_individual_neurons(neuron_seg_file):
    with h5py.File(neuron_seg_file, 'r') as f:
        neuron_seg = f['labels'][:]

    dict1 = (segid, coords)
    dict2 = (segid, num of vesicles)

    for each segid in neuron_seg:
        #code from load_calculate_and_visualize_neuron_and_vesicles -> init dict2
'''



#vesicle density heatmap
def calculate_density_map(labeled_vesicles):
    #overlay vesicles
    print("calculate_density_map running")

    #simple overlay
    density_map = np.zeros(labeled_vesicles.shape, dtype=np.float32)
    for label_num in np.unique(labeled_vesicles):
        if label_num == 0:
            continue
        density_map[labeled_vesicles == label_num] += 1

    print("shape of vesicles density map: ", density_map.shape) #(188, 4096, 4096)
    return density_map

    
    '''
    density_map = np.zeros(labeled_vesicles.shape, dtype=np.float32)
    
    for x, y, z in vesicle_coords: #zyx?
        if 0 <= y <= shape[2] and 0 <= x < shape[1] and 0 <= y < shape[0]:
            density_map[int(y), int(x)] += 1
    
    #gaussian
    density_map = gaussian_filter(density_map, sigma=sigma)
    
    return density_map
    '''

    #return as a zarr array? -> neuroglancer

#vesicles_within_perimeter_labels
def visualize_heatmap(volume, neuron_data):

    viewer = napari.Viewer()
    
    viewer.add_image(neuron_data, colormap='gray', name='bg', contrast_limits=[0, np.max(neuron_data)])
    viewer.add_image(volume, colormap='RdBu', contrast_limits=[0, np.max(volume)], blending = 'additive')

    napari.run()


