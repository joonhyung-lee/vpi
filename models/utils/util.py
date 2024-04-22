import re
from IPython.display import Markdown,display
import math,time,os
import numpy as np
import shapely as sp # handle polygon
from shapely.geometry import Polygon,LineString,Point # handle polygons
from scipy.spatial.distance import cdist
import re
from IPython.display import Markdown,display

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def r2t(R):
    """
        Convert rotation matrix to transformation matrix
    """
    # Initialize a 4x4 identity matrix
    T = np.eye(4)
    
    # Set the top-left 3x3 block to the input rotation matrix
    T[:3, :3] = R
    
    return T

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R    

def rpy2r(rpy_rad):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2quat(R):
    """ 
        Convert Rotation Matrix to Quaternion.  See rotation.py for notes 
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def skew(x):
    """ 
        Get a skew-symmetric matrix
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        Compute the rotation matrix from an angular velocity vector
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] norm of a should be 1.0 not [%.2e]."%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R
    
def np_uv(vec):
    """
        Get unit vector
    """
    x = np.array(vec)
    return x/np.linalg.norm(x)

def get_rotation_matrix_from_two_points(p_fr,p_to):
    p_a  = np.copy(np.array([0,0,1]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R
    

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def is_point_in_polygon(point,polygon):
    """
        Is the point inside the polygon
    """
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
        Is the point feasible w.r.t. obstacle list
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False

def is_point_to_point_connectable(point1,point2,obs_list):
    """
        Is the line connecting two points connectable
    """
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False
    
class TicTocClass(object):
    """
        Tic toc
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            Initialize
        """
        self.name        = name
        self.time_start  = time.time()
        self.time_end    = time.time()
        self.print_every = print_every

    def tic(self):
        """
            Tic
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=0,VERBOSE=True):
        """
            Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if VERBOSE:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if (cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))

def get_interp_const_vel_traj(traj_anchor,vel=1.0,HZ=100,ord=np.inf):
    """
        Get linearly interpolated constant velocity trajectory
    """
    L = traj_anchor.shape[0]
    D = traj_anchor.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = traj_anchor[tick-1,:],traj_anchor[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    traj_interp = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D):
        traj_interp[:,d_idx] = np.interp(times_interp,times_anchor,traj_anchor[:,d_idx])
    return times_interp,traj_interp

def meters2xyz(depth_img,cam_matrix):
    """
        Scaled depth image to pointcloud
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def compute_view_params(camera_pos,target_pos,up_vector=np.array([0,0,1])):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    iter_tick = 0
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
            iter_tick += 1
            if iter_tick > 1000:
                print ("[sample_xyzs] iter_tick:[%d] is over 1000."%(iter_tick))
                break
        xyzs[p_idx,:] = xyz
    return xyzs

def create_folder_if_not_exists(file_path):
    """ 
        Create folder if not exist
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print ("[%s] created."%(folder_path))
        
def quintic_trajectory(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, duration, num_points, max_velocity, max_acceleration):
    # Compute time intervals
    t = np.linspace(0, duration, num_points)

    # Compute polynomial coefficients for each joint
    joint_coeffs = []
    for i in range(6):
        # Solve for polynomial coefficients: Ax = b (A=quintic trajectory, x=coeff vec, b=constraints)
        # A: 6x6 matrix, x: 6x1 vector, b: 6x1 vector
        A = np.array([[0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 2, 0, 0],
                      [duration**5, duration**4, duration**3, duration**2, duration, 1],
                      [5*duration**4, 4*duration**3, 3*duration**2, 2*duration, 1, 0],
                      [20*duration**3, 12*duration**2, 6*duration, 2, 0, 0]])

        b = np.array([start_pos[i], start_vel[i], start_acc[i], end_pos[i], end_vel[i], end_acc[i]])

        # Solve for polynomial coefficients
        x = np.linalg.solve(A, b)
        joint_coeffs.append(x)

    # Generate trajectory points for each joint
    positions = np.zeros((num_points, 6))
    velocities = np.zeros((num_points, 6))
    accelerations = np.zeros((num_points, 6))
    jerks = np.zeros((num_points, 6))

    for i in range(num_points):
        for j in range(6):
            positions[i, j] = np.polyval(joint_coeffs[j], t[i])   # Compute 0th derivative
            velocities[i, j] = np.polyval(np.polyder(joint_coeffs[j]), t[i])  # Compute 1st derivative
            accelerations[i, j] = np.polyval(np.polyder(np.polyder(joint_coeffs[j])), t[i])   # Compute 2nd derivative
            jerks[i, j] = np.polyval(np.polyder(np.polyder(np.polyder(joint_coeffs[j]))), t[i])   # Compute 3rd derivative

    # Apply velocity and acceleration constraints
    velocities = np.clip(velocities, -max_velocity, max_velocity)
    accelerations = np.clip(accelerations, -max_acceleration, max_acceleration)

    return positions, velocities, accelerations, jerks

# passthrough filter about specific axis
def passthrough_filter(pcd, axis, interval):
    mask = (pcd[:, axis] > interval[0]) & (pcd[:, axis] < interval[1])
    return pcd[mask]

def remove_duplicates_with_threshold(pointcloud, threshold=0.05):
    filtered_pointcloud = []

    for point in pointcloud:
        # Check if the point is close to any point already in the filtered list
        if all(np.linalg.norm(np.array(point) - np.array(existing_point)) > threshold for existing_point in filtered_pointcloud):
            filtered_pointcloud.append(point)

    return np.array(filtered_pointcloud)

def printmd(string):
    display(Markdown(string))
    
def extract_quoted_words(string):
    quoted_words = re.findall(r'"([^"]*)"', string)
    return quoted_words    

# Wine / Shelves
def get_geom_region_type1(env, obj_name):
    center_x, center_y, center_z, size_w, size_h, size_d = env.model.geom_aabb[env.model.body(obj_name).geomadr][0]
    obj_pos_x, obj_pos_y, obj_pos_z = env.get_p_body(obj_name)
    obj_R = env.get_R_body(obj_name)
    
    # Get the corners of the object based on the lower corner
    corners = np.array([
        # [center_x - size_w, center_y - size_h, center_z - size_d],
        [center_x - size_w, center_y - size_h, center_z + size_d],
        # [center_x - size_w, center_y + size_h, center_z - size_d],
        [center_x - size_w, center_y + size_h, center_z + size_d],
        # [center_x + size_w, center_y - size_h, center_z - size_d],
        [center_x + size_w, center_y - size_h, center_z + size_d],
        # [center_x + size_w, center_y + size_h, center_z - size_d],
        [center_x + size_w, center_y + size_h, center_z + size_d],
    ])
    
    # Consider Rotation and Translation of the obj_name
    corners = np.matmul(corners, obj_R.T) + np.array([obj_pos_x, obj_pos_y, obj_pos_z])

    return corners

# Books / Cylinder
def get_geom_region_type2(env, obj_name):
    center_x, center_y, center_z, size_w, size_h, size_d = env.model.geom_aabb[env.model.body(obj_name).geomadr][0]
    # size_w, size_h, size_d = env.model.geom_size[env.model.body(obj_name).geomadr][0]
    obj_pos_x, obj_pos_y, obj_pos_z = env.get_p_body(obj_name)
    obj_R = env.get_R_body(obj_name)
    
    # Get the corners of the object
    corners = np.array([
        [center_x - size_h, center_y - size_w/2, center_z],
        [center_x - size_h, center_y - size_w/2, center_z + 2*size_d],
        [center_x - size_h, center_y + size_w/2, center_z],
        [center_x - size_h, center_y + size_w/2, center_z + 2*size_d],
        [center_x + size_h, center_y - size_w/2, center_z],
        [center_x + size_h, center_y - size_w/2, center_z + 2*size_d],
        [center_x + size_h, center_y + size_w/2, center_z],
        [center_x + size_h, center_y + size_w/2, center_z + 2*size_d],
    ])
    # Consider Rotation and Translation of the obj_name
    corners = np.matmul(corners, obj_R.T) + np.array([obj_pos_x, obj_pos_y, obj_pos_z])

    return corners

# Avoiding Table / Mug cup
def get_geom_region_type3(env, obj_name):
    center_x, center_y, center_z, size_w, size_h, size_d = env.model.geom_aabb[env.model.body(obj_name).geomadr][0]
    # size_w, size_h, size_d = env.model.geom_size[env.model.body(obj_name).geomadr][0]
    obj_pos_x, obj_pos_y, obj_pos_z = env.get_p_body(obj_name)
    obj_R = env.get_R_body(obj_name)
    
    # Get the corners of the object
    corners = np.array([
        # [center_x - size_w, center_y - size_h, center_z],
        [center_x - size_w, center_y - size_h*0.85, center_z + 2*size_d],
        # [center_x - size_w, center_y + size_h, center_z],
        [center_x - size_w, center_y + size_h*0.85, center_z + 2*size_d],
        # [center_x + size_w, center_y - size_h, center_z],
        [center_x + size_w, center_y - size_h*0.85, center_z + 2*size_d],
        # [center_x + size_w, center_y + size_h, center_z],
        [center_x + size_w, center_y + size_h*0.85, center_z + 2*size_d],
    ])
    # Consider Rotation and Translation of the obj_name
    corners = np.matmul(corners, obj_R.T) + np.array([obj_pos_x, obj_pos_y, obj_pos_z])

    return corners

# Wine / Shelves
def get_geom_region_type4(env, obj_name):
    center_x, center_y, center_z, size_w, size_h, size_d = env.model.geom_aabb[env.model.body(obj_name).geomadr][0]
    obj_pos_x, obj_pos_y, obj_pos_z = env.get_p_body(obj_name)
    obj_R = env.get_R_body(obj_name)
    
    # Get the corners of the object based on the lower corner
    corners = np.array([
        # [center_x - size_w, center_y - size_h, center_z - size_d],
        [center_x - size_w, center_y - size_h, center_z + size_d],
        # [center_x - size_w, center_y + size_h, center_z - size_d],
        [center_x - size_w, center_y + size_h, center_z + size_d],
        # [center_x + size_w, center_y - size_h, center_z - size_d],
        [center_x + size_w, center_y - size_h, center_z + size_d],
        # [center_x + size_w, center_y + size_h, center_z - size_d],
        [center_x + size_w, center_y + size_h, center_z + size_d],
    ])
    
    # Consider Rotation and Translation of the obj_name
    corners = np.matmul(corners, obj_R.T) + np.array([obj_pos_x, obj_pos_y, obj_pos_z])

    return corners

def fill_object(vertices, resolution):
    # Assuming vertices is a list of 8 vertices of an axis-aligned box
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)

    # Creating points using linspace
    x_points = np.linspace(min_x, max_x, resolution[0])
    y_points = np.linspace(min_y, max_y, resolution[1])
    z_points = np.linspace(min_z, max_z, resolution[2])

    # Creating a meshgrid for the points
    x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')

    # Reshaping to get a list of points
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    return points

# passthrough filter about specific axis
def passthrough_filter(pcd, axis, interval):
    mask = (pcd[:, axis] > interval[0]) & (pcd[:, axis] < interval[1])
    return pcd[mask]

def remove_duplicates_with_threshold(pointcloud, threshold=0.05):
    filtered_pointcloud = []

    for point in pointcloud:
        # Check if the point is close to any point already in the filtered list
        if all(np.linalg.norm(np.array(point) - np.array(existing_point)) > threshold for existing_point in filtered_pointcloud):
            filtered_pointcloud.append(point)

    return np.array(filtered_pointcloud)

def remove_duplicates_with_specific_com(p_com, pointcloud, threshold=0.05):
    filtered_pointcloud = []
    
    for point in pointcloud:
        # Check if the point is close to the specific center of mass (p_com)
        if np.linalg.norm(np.array(point) - np.array(p_com)) < threshold:
            filtered_pointcloud.append(point)
            
    return np.array(filtered_pointcloud)

def downsample_pointcloud(pointcloud, grid_size):
    # Determine the range of values for each coordinate
    min_vals = pointcloud.min(axis=0)
    max_vals = pointcloud.max(axis=0)

    # Normalize the pointcloud to a grid
    grid_pointcloud = np.floor((pointcloud - min_vals) / grid_size).astype(int)

    # Use a dictionary to store one point per grid cell
    unique_pointcloud = {}
    for pos, original_pos in zip(grid_pointcloud, pointcloud):
        pos_tuple = tuple(pos)
        if pos_tuple not in unique_pointcloud:
            unique_pointcloud[pos_tuple] = original_pos

    # Extract the downsampled pointcloud
    downsampled_pointcloud = np.array(list(unique_pointcloud.values()))

    return downsampled_pointcloud

def make_ordinal(n):
    '''
        Convert an integer into its ordinal representation::
            make_ordinal(0)   => '0th'
            make_ordinal(3)   => '3rd'
            make_ordinal(122) => '122nd'
            make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix
