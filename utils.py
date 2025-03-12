from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import torch 

def uniformize(curves: torch.tensor, n: int = 200) -> torch.tensor:
    with torch.no_grad():
        l = torch.cumsum(torch.nn.functional.pad(torch.norm(curves[:,1:,:] - curves[:,:-1,:],dim=-1),[1,0,0,0]),-1)
        l = l/l[:,-1].unsqueeze(-1)
        
        sampling = torch.linspace(0,1,n).to(l.device).unsqueeze(0).tile([l.shape[0],1])
        end_is = torch.searchsorted(l,sampling)[:,1:]
        end_ids = end_is.unsqueeze(-1).tile([1,1,2])
        
        l_end = torch.gather(l,1,end_is)
        l_start = torch.gather(l,1,end_is-1)
        ws = (l_end - sampling[:,1:])/(l_end-l_start)
    
    end_gather = torch.gather(curves,1,end_ids)
    start_gather = torch.gather(curves,1,end_ids-1)
    
    uniform_curves = torch.cat([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws.unsqueeze(-1))],1)

    return uniform_curves

def preprocess_curves(curves: torch.tensor, n: int = 200) -> torch.tensor:
    
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves,n)

    # center curves
    curves = curves - curves.mean(1).unsqueeze(1)
    
    # apply uniform scaling
    s = torch.sqrt(torch.square(curves).sum(-1).sum(-1)/n).unsqueeze(-1).unsqueeze(-1)
    curves = curves/s

    # find the furthest point on the curve
    max_idx = torch.square(curves).sum(-1).argmax(dim=1)

    # rotate curves so that the furthest point is horizontal
    # theta = -torch.atan2(curves[torch.arange(curves.shape[0]),max_idx,1],curves[torch.arange(curves.shape[0]),max_idx,0])
    theta = torch.rand([curves.shape[0]]).to(curves.device) * 2 * np.pi

    # normalize the rotation
    R = torch.eye(2).unsqueeze(0).to(curves.device)
    R = R.repeat([curves.shape[0],1,1])

    R[:,0,0] = torch.cos(theta)
    R[:,0,1] = -torch.sin(theta)
    R[:,1,0] = torch.sin(theta)
    R[:,1,1] = torch.cos(theta)

    curves = torch.bmm(R,curves.transpose(1,2)).transpose(1,2)

    return curves

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    
    # Remove the 'module.' prefix if present
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load the modified state_dict
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model weights loaded from {checkpoint_path}")
    return model

def create_adjacency_matrix(edges):
    # Get the maximum node index to define matrix size
    max_node = edges.max().item()
    
    # Initialize a square matrix of size (max_node+1) x (max_node+1) with False
    adj_matrix = np.full((max_node + 1, max_node + 1), False, dtype=bool)
    
    # Fill the adjacency matrix with True for each edge
    for i in range(edges.shape[1]):
        start_node, end_node = edges[0, i].item(), edges[1, i].item()
        adj_matrix[start_node, end_node] = True
        adj_matrix[end_node, start_node] = True  # Symmetric for undirected graph
    
    return adj_matrix

def truncate_at_closest_with_tolerance(array, reference_point=np.array([1.0, 1.0]), tolerance=0.05):
    """
    Finds the closest point to reference_point within a given tolerance and truncates 
    the array before the first occurrence of a point within that tolerance.
    """
    distances = np.linalg.norm(array - reference_point, axis=1)
    min_distance = np.min(distances)  # Get the closest distance

    # Find all points that are within the tolerance range of the closest point
    valid_indices = np.where(distances <= min_distance + tolerance)[0]
    
    if len(valid_indices) == 0:
        return array  # No valid points, return original array

    # Get the first occurrence of a point within the tolerance range
    first_valid_index = valid_indices[0]

    return array[:first_valid_index]  # Keep all points from this index onward



def run_imap_multiprocessing(func, argument_list, show_prog = True):
    pool = mp.Pool(processes=mp.cpu_count())
    
    if show_prog:            
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list),position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.imap(func=func, iterable=argument_list):
            result_list_tqdm.append(result)

    return result_list_tqdm

# class curve_normalizer():
#     def __init__(self, scale=True):
#         """Intance of curve rotation and scale normalizer.
#         Parameters
#         ----------
#         scale: boolean
#                 If true curves will be oriented and scaled to the range of [0,1]. Default: True.
#         """
#         self.scale = scale
#         self.vfunc = np.vectorize(lambda c: self.get_oriented(c),signature='(n,m)->(n,m)')
        
#     def get_oriented(self, curve):
#         """Orient and scale(if enabled on initialization) the curve to the normalized configuration
#         Parameters
#         ----------
#         curve: [n_point,2]
#                 Point coordinates describing the curve.

#         Returns
#         -------
#         output curve: [n_point,2]
#                 Point coordinates of the curve oriented such that the maximum length is parallel to the x-axis and 
#                 scaled to have exactly a width of 1.0 on the x-axis is scale is enabled. Curve position is also 
#                 normalized to be at x=0 for the left most point and y=0 for the bottom most point.
#         """
#         ci = 0
#         t = curve.shape[0]
#         pi = t
        
#         while pi != ci:
#             pi = t
#             t = ci
#             ci = np.argmax(np.linalg.norm(curve-curve[ci],2,1))
        
#         d = curve[pi] - curve[t]
        
#         if d[1] == 0:
#             theta = 0
#         else:
#             d = d * np.sign(d[1])
#             theta = -np.arctan(d[1]/d[0])
        
#         rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
#         out = np.matmul(rot,curve.T).T
#         out = out - np.min(out,0)
        
#         rot2 = np.array([[np.cos(theta+np.pi),-np.sin(theta+np.pi)],[np.sin(theta+np.pi),np.cos(theta+np.pi)]])
#         out2 = np.matmul(rot2,curve.T).T
#         out2 = out2 - np.min(out2,0)
        
#         m1 = out[np.abs(out[:,0] - 0.5).argsort()[0:5],1].max()
#         m2 = out2[np.abs(out2[:,0] - 0.5).argsort()[0:5],1].max()
        
#         if m2<m1:
#             out = out2
        
#         if self.scale:
#             out = out/np.max(out,0)[0]
        
#         if np.isnan(np.sum(out)):
#             out = np.zeros(out.shape)
                    
#         return out
    
#     def __call__(self, curves):
#         """Orient and scale(if enabled on initialization) the batch of curve to the normalized configuration
#         Parameters
#         ----------
#         curve: [N,n_point,2]
#                 batch of point coordinates describing the curves.

#         Returns
#         -------
#         output curve: [N,n_point,2]
#                 Batch of point coordinates of the curve oriented such that the maximum length is parallel to the x-axis and 
#                 scaled to have exactly a width of 1.0 on the x-axis is scale is enabled. Curve position is also 
#                 normalized to be at x=0 for the left most point and y=0 for the bottom most point.
#         """
#         return self.vfunc(curves)