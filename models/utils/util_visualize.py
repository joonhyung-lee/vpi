import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
# import pyvista as pv
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from itertools import combinations
import math
from matplotlib import gridspec

def add_padding(img, target_width):
    height, width = img.shape[:2]
    padding_left = (target_width - width) // 2
    padding_right = target_width - width - padding_left
    return cv2.copyMakeBorder(img, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT)

def visualize_subplots(images, cols=3):
    """
    Visualize a list of images.

    Parameters:
    images (list): List of images. Each image can be a path to an image file or a numpy array.
    cols (int): Number of columns in the image grid.
    """
    imgs_to_show = []

    # Load images if they are paths, or directly use them if they are numpy arrays
    for img in images:
        if isinstance(img, str):  # Assuming it's a file path
            img_data = plt.imread(img)
        elif isinstance(img, np.ndarray):  # Assuming it's a numpy array
            img_data = img
        else:
            raise ValueError("Images should be either file paths or numpy arrays.")
        imgs_to_show.append(img_data)

    N = len(imgs_to_show)
    if N == 0:
        print("No images to display.")
        return

    rows = int(math.ceil(N / cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(cols * 4, rows * 4))

    for n in range(N):
        ax = fig.add_subplot(gs[n])
        ax.imshow(imgs_to_show[n])
        ax.set_title(f"Image {n + 1}")
        ax.axis('off')

    fig.tight_layout()
    plt.show()
def show_images(before_scene, before_rgb, before_paired_img,
                after_scene, after_rgb, after_paired_img, title="Before and After Interaction"):
    # Calculate the resize dimensions for scene and rgb images
    dsize = (before_rgb.shape[1] // 4, before_rgb.shape[0] // 4)
    # Calculate the square size for paired images
    square_size = min(before_rgb.shape[:2]) // 3
    square_dsize = (square_size, square_size)

    # Convert and resize before images
    before_scene = cv2.cvtColor(before_scene, cv2.COLOR_BGR2RGB)
    before_rgb = cv2.cvtColor(before_rgb, cv2.COLOR_BGR2RGB)
    before_scene_img_resized = cv2.resize(before_scene, dsize)
    before_rgb_img_resized = cv2.resize(before_rgb, dsize)

    # Resize paired image to a square and add padding to match the width of scene and rgb images
    before_paired_img = cv2.cvtColor(before_paired_img, cv2.COLOR_BGR2RGB)
    before_paired_img_resized = cv2.resize(before_paired_img, square_dsize)
    padded_before_paired_img = add_padding(before_paired_img_resized, dsize[0])

    # Stack the before images vertically, ensuring all have the same width
    before_combined = np.vstack((before_scene_img_resized, before_rgb_img_resized, padded_before_paired_img))

    # Repeat for after images if they exist
    if after_scene is not None and after_rgb is not None and after_paired_img is not None:
        after_scene = cv2.cvtColor(after_scene, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(after_rgb, cv2.COLOR_BGR2RGB)
        after_scene_img_resized = cv2.resize(after_scene, dsize)
        after_rgb_img_resized = cv2.resize(after_rgb, dsize)

        after_paired_img = cv2.cvtColor(after_paired_img, cv2.COLOR_BGR2RGB)
        after_paired_img_resized = cv2.resize(after_paired_img, square_dsize)
        padded_after_paired_img = add_padding(after_paired_img_resized, dsize[0])

        after_combined = np.vstack((after_scene_img_resized, after_rgb_img_resized, padded_after_paired_img))
        all_combined = np.hstack((before_combined, after_combined))
    else:
        all_combined = before_combined

    # Display the combined image
    cv2.imshow(title, all_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def show_point_cloud_with_pyvista(point_cloud):
#     """
#     Visualize a point cloud using PyVista.

#     Args:
#         point_cloud (np.ndarray): The coordinates of the point cloud.
#     """
#     # Create a PyVista plotter
#     plotter = pv.Plotter()

#     # Create a point cloud mesh from the numpy array
#     cloud_mesh = pv.PolyData(point_cloud)

#     # Add the point cloud mesh to the plotter
#     plotter.add_mesh(cloud_mesh, point_size=5)

#     # Show the plot
#     plotter.show()


def show_point_cloud(point_cloud, axis=False, title='Point Cloud', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis'):
    """
    Visualize a point cloud.

    Args:
        point_cloud (np.ndarray): The coordinates of the point cloud.
        axis (bool, optional): Hide the coordinate of the matplotlib. Defaults to False.
        title (str, optional): Title of the plot. Defaults to 'Point Cloud'.
        xlabel (str, optional): Label for the X-axis. Defaults to 'X-axis'.
        ylabel (str, optional): Label for the Y-axis. Defaults to 'Y-axis'.
        zlabel (str, optional): Label for the Z-axis. Defaults to 'Z-axis'.
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2], s=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def show_point_clouds(point_clouds, axis=False, device='cuda'):
    """visual a point cloud
    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False.
    """
    ax = plt.figure().add_subplot(projection='3d')
    for idx, point_cloud in enumerate(point_clouds):
        pcd_np = np.array(point_cloud)
        pcd_torch = torch.from_numpy(pcd_np).permute([1,0]).unsqueeze(0).to(device)
        ax.scatter(xs=pcd_torch.cpu().detach().numpy()[0, :, 0], ys=pcd_torch.cpu().detach().numpy()[0, :, 1], zs=pcd_torch.cpu().detach().numpy()[0, :, 2], s=5)
    ax._axis3don = False
    plt.show()

def setup_seed(seed):
    """
    Set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# passthrough filter about specific axis
def passthrough_filter(pcd, axis, interval):
    mask = (pcd[:, axis] > interval[0]) & (pcd[:, axis] < interval[1])
    return pcd[mask]

def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.

    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_

def euclidean_distance_3d(coord1, coord2):
    """Calculate the Euclidean distance between two points in 3D."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

def normalize_values(values, min_value=0.1, max_value=1.0):
    min_dist = min(values)
    max_dist = max(values)
    if min_dist == max_dist:
        return [max_value] * len(values)
    return [max(min_value, min(min_value + (max_value - min_value) * (value - min_dist) / (max_dist - min_dist), max_value)) for value in values]

# def find_closest_objects_3d(coordinates: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
#     """
#         Find the closest object for each object in 3D space based on their coordinates.
#     """
#     closest_pairs = []

#     for i in range(len(coordinates)):
#         min_distance = float('inf')
#         closest_object = -1
#         for j in range(len(coordinates)):
#             if i != j:
#                 distance = euclidean_distance_3d(coordinates[i], coordinates[j])
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_object = j
#         closest_pairs.append((i, closest_object))

#     return closest_pairs

def find_closest_objects_3d(coordinates: List[Tuple[float, float, float]], object_names: List[str], min_distance: float = 0.1) -> Tuple[List[Tuple[int, int]], List[Tuple[str, str]]]:
    closest_pairs_indices = []
    closest_pairs_names = []

    for i in range(len(coordinates)):
        closest_object_index = -1
        min_found_distance = float('inf')
        for j in range(len(coordinates)):
            if i != j:
                distance = euclidean_distance_3d(coordinates[i], coordinates[j])
                # print(f'{object_names[i]} and {object_names[j]}: {distance:.2f}')
                if 0 < distance < min_found_distance:
                    min_found_distance = distance
                    closest_object_index = j

        # Check if a close enough object was found
        if min_found_distance < min_distance:
            closest_pairs_indices.append((i, closest_object_index))
            closest_pairs_names.append((object_names[i], object_names[closest_object_index]))

    return closest_pairs_indices, closest_pairs_names

def get_closest_pairs_img(coordinates: List[Tuple[float, float, float]], 
                          pairs: List[Tuple[int, int]],
                          labels: List[str],
                          TOP_VIEW: bool) -> np.ndarray:
    fig = Figure()
    ax = fig.add_subplot(111, projection='3d')

    # Swap coordinates if needed
    swapped_coordinates = [(y, x, z) if TOP_VIEW else (x, y, z) for x, y, z in coordinates]

    # Calculate distances and normalize for alpha values
    distances = [euclidean_distance_3d(coord1, coord2) for coord1, coord2 in combinations(coordinates, 2)]
    normalized_alpha = normalize_values(distances)

    # Plot nodes and labels
    for coord, label in zip(swapped_coordinates, labels):
        ax.scatter(*coord, color='black')
        ax.text(*coord, label)

    # Plot edges with varying alpha values
    for ((i, coord1), (j, coord2)), alpha in zip(combinations(enumerate(swapped_coordinates), 2), normalized_alpha):
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color='gray', alpha=alpha)

    # Different colors for each pair
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    for pair, color in zip(pairs, colors):
        coord1 = swapped_coordinates[pair[0]]
        coord2 = swapped_coordinates[pair[1]]
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color=color)

    # Set axis labels and limits
    ax.set_xlim(-0.38, 0.38)    # Labelled as Y Axis (Originally X)
    ax.set_ylim(0.525, 1.15)    # Labelled as X Axis (Originally Y)
    ax.set_zlim(0.65, 0.85)
    ax.set_xlabel('X Axis' if not TOP_VIEW else 'Y Axis')
    ax.set_ylabel('Y Axis' if not TOP_VIEW else 'X Axis')
    ax.set_zlabel('Z Axis')
    # Invert the Y axis
    ax.invert_yaxis()

    if TOP_VIEW:
        # ax.view_init(elev=75, azim=90)
        ax.view_init(elev=85, azim=90)
        mid_z = (coordinates[0][2] + coordinates[-1][2]) / 2  # Midpoint of z-axis, adjust as needed
        ax.set_zticks([mid_z])       # Shows only one tick at the midpoint
        ax.set_zticklabels([f'{mid_z:.1f}'])  # Optionally set a custom label

    # Convert plot to image
    plot_image = plot_to_image(fig)
    return plot_image

def plot_to_image(fig, dpi=400):
    """
        Convert a Matplotlib figure to a 3-channel RGB image in NumPy array format, with increased resolution.
    """
    buf = BytesIO()
    # Save the plot as a PNG in memory with high DPI and minimized padding
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    
    # Convert PNG buffer to NumPy array
    img_arr = np.array(Image.open(buf))
    
    # Convert RGB to BGR format (for OpenCV)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    
    buf.close()
    
    return img_arr

# def get_closest_pairs_img(coordinates: List[Tuple[float, float, float]], 
#                           pairs: List[Tuple[int, int]], 
#                           labels: List[str],
#                           TOP_VIEW: bool) -> np.ndarray:
#     """
#     Generate an image of the 3D plot showing the closest pairs of objects with their labels,
#     with swapped X and Y axes, using the object-oriented approach of matplotlib.
#     """
#     fig = Figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Swap X and Y coordinates for each point
#     swapped_coordinates = [(y, x, z) for x, y, z in coordinates]

#     # Plot each coordinate as a black dot and label it
#     for coord, label in zip(swapped_coordinates, labels):
#         ax.scatter(*coord, color='black')
#         ax.text(coord[0], coord[1], coord[2], label)

#     # Different colors for each pair
#     colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

#     # Draw lines between each pair
#     for pair, color in zip(pairs, colors):
#         coord1 = swapped_coordinates[pair[0]]
#         coord2 = swapped_coordinates[pair[1]]
#         ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color=color)
# 
#     # Set axis labels and limits
#     ax.set_xlim(-0.38, 0.38)    # Labelled as Y Axis (Originally X)
#     ax.set_ylim(0.525, 1.15)    # Labelled as X Axis (Originally Y)
#     ax.set_zlim(0.65, 0.85)

#     ax.set_xlabel('Y Axis', labelpad=5)  # Labelled as Y Axis (Originally X)
#     ax.set_ylabel('X Axis', labelpad=5)  # Labelled as X Axis (Originally Y)
#     ax.set_zlabel('Z Axis', labelpad=5)

#     # Invert the Y axis
#     ax.invert_yaxis()

#     if TOP_VIEW:
#         # ax.view_init(elev=75, azim=90)
#         ax.view_init(elev=85, azim=90)
#         mid_z = (coordinates[0][2] + coordinates[-1][2]) / 2  # Midpoint of z-axis, adjust as needed
#         ax.set_zticks([mid_z])       # Shows only one tick at the midpoint
#         ax.set_zticklabels([f'{mid_z:.1f}'])  # Optionally set a custom label
#     plot_image = plot_to_image(fig)

#     return plot_image

# def plot_closest_pairs(coordinates: List[Tuple[float, float, float]], pairs: List[Tuple[int, int]]):
#     """
#         Plot the 3D coordinates of objects and connect each object to its closest pair with a line.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot each coordinate as a black dot
#     for coord in coordinates:
#         ax.scatter(*coord, color='black')

#     # Different colors for each pair
#     colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

#     # Draw lines between each pair
#     for pair, color in zip(pairs, colors):
#         coord1 = coordinates[pair[0]]
#         coord2 = coordinates[pair[1]]
#         ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color=color)

#     # Adjust labels and tick marks to prevent overlapping
#     ax.set_xlabel('X Axis', labelpad=10)
#     ax.set_ylabel('Y Axis', labelpad=10)
#     ax.set_zlabel('Z Axis', labelpad=10)
#     ax.tick_params(axis='x', pad=5)
#     ax.tick_params(axis='y', pad=5)
#     ax.tick_params(axis='z', pad=5)

#     plt.title('Closest Object Pairs')
#     plt.tight_layout()
#     plt.show()

# def plot_closest_pairs_with_labels(coordinates: List[Tuple[float, float, float]], 
#                                    pairs: List[Tuple[int, int]], 
#                                    labels: List[str]):
#     """
#         Plot the 3D coordinates of objects, connect each object to its closest pair with a line, 
#         and label each object with its name.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot each coordinate as a black dot and label it
#     for coord, label in zip(coordinates, labels):
#         ax.scatter(*coord, color='black')
#         ax.text(coord[0], coord[1], coord[2], label)

#     # Different colors for each pair
#     colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

#     # Draw lines between each pair
#     for pair, color in zip(pairs, colors):
#         coord1 = coordinates[pair[0]]
#         coord2 = coordinates[pair[1]]
#         ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color=color)

#     # Adjust labels and tick marks to prevent overlapping
#     ax.set_xlabel('X Axis', labelpad=10)
#     ax.set_ylabel('Y Axis', labelpad=10)
#     ax.set_zlabel('Z Axis', labelpad=10)
#     ax.tick_params(axis='x', pad=5)
#     ax.tick_params(axis='y', pad=5)
#     ax.tick_params(axis='z', pad=5)

#     plt.title('Closest Object Pairs with Labels')
#     plt.tight_layout()
#     plt.show()

if __name__ == '__main__':
    pcs = torch.rand(32, 3, 1024)
    knn_index = knn(pcs, 16)
    print(knn_index.size())
    knn_pcs = index_points(pcs.permute(0, 2, 1), knn_index)
    print(knn_pcs.size())