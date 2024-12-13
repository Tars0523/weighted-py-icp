import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
import open3d as o3d

def rotation_matrix_from_rotvec(rotvec):
    theta = torch.linalg.norm(rotvec)
    if theta < 1e-12:
        return torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    k = rotvec / theta
    K = torch.tensor([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], dtype=rotvec.dtype, device=rotvec.device)
    R = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device) + torch.sin(theta)*K + (1-torch.cos(theta))*(K@K)
    return R

class WeightedICP:
    def __init__(self):
        self.device = 'cpu'
        self.src = None
        self.tgt = None
        
        self.src_correspondence = None
        self.tgt_correspondence = None
        self.normals = None
        self.correspondence_distances = None
        self.correspondence_indices = None

        self.alpha = None
        self.max_corr_distance = None
        self.k = None
        self.planar_threshold = None

        self.src_np = None
        self.tgt_np = None
        self.tree = None

        self.planar_mask = None
        
        
    def compute_normals(self):
        
        _, indices = self.tree.query(self.tgt_np, k=self.k)  # indices: (M', k)

       
        indices_t = torch.from_numpy(indices).to(self.device) # (M', k)
        tgt_tensor = torch.from_numpy(self.tgt_np).to(self.device, dtype=torch.float32) # (M', 3)

        neighbors = tgt_tensor[indices_t] # (M', k, 3)
        
        centroids = neighbors.mean(dim=1, keepdim=True) # calculate centroids (M', 1, 3)

        centered_neighbors = neighbors - centroids  # dodo center (M', k, 3)
        
        cov_matrices = torch.matmul(centered_neighbors.transpose(1,2), centered_neighbors) / self.k # compute covariance matrix (M', 3, 3)

        eigvals, eigvecs = torch.linalg.eigh(cov_matrices) # Eigen Value Decompose (M',3), (M',3,3), cholesky more fatser less accurate
        
        normals = eigvecs[:,:,0] # noraml vector (M',3)

        norm = torch.norm(normals, dim=1, keepdim=True) # (M', 1)
        mask_nonzero = (norm > 1e-12).view(-1) # (M',)
        normals[mask_nonzero] = normals[mask_nonzero] / norm[mask_nonzero].view(-1,1) # (M',3)

        self.normals = normals # (M',3)

    def nearest_neighbor(self):
        distances, indices = self.tree.query(self.src_np, k=1) # distance: N' , indicies : N' 
        distances = torch.from_numpy(distances).to(self.device, dtype=torch.float32) # N' 
        indices = torch.from_numpy(indices).to(self.device, dtype=torch.int64) # N' 
        mask = distances < self.max_corr_distance # N''
        self.src_correspondence = self.src[mask]  # N''
        self.tgt_correspondence = self.tgt[indices[mask]] # N''
        self.correspondence_distances = distances[mask] # N''
        self.correspondence_indices = indices[mask] # N''

    def normal_candidate(self):
        
        _, indices = self.tree.query(self.tgt_np, k=self.k)

        neighbors = self.tgt_np[indices]  # M', k, 3
        centroids = neighbors.mean(axis=1, keepdims=True)  # M', 1, 3
        centered_neighbors = neighbors - centroids  # M', k, 3
        
        centered_neighbors = torch.tensor(centered_neighbors, device=self.device, dtype=torch.float32)  # M', k, 3
        cov_matrices = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors) / self.k  # M', 3, 3

        U, S, Vh = torch.linalg.svd(cov_matrices)  # S: (M', 3), U: (M', 3, 3), Vh: (M', 3, 3)
        
        # λ3 / (λ1 + λ2 + λ3)
        lambda1 = S[:, 0]
        lambda2 = S[:, 1]
        lambda3 = S[:, 2]
        surface_variation = lambda3 / (lambda1 + lambda2 + lambda3 + 1e-12) # (M',)

        # Determine planar regions: surface_variation < planar_threshold
        planar_mask = (surface_variation < self.planar_threshold).int() # (M',)

        self.planar_mask = planar_mask # (M',)

    def downsample_with_open3d(self, points, voxel_size):
        pcd = o3d.geometry.PointCloud()
        points_np = points.cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return torch.from_numpy(np.asarray(downsampled_pcd.points)).to(self.device, dtype=torch.float32)

    def run(self, src, tgt, 
            alpha = 0.5, max_corr_distance = 0.3, 
            k = 10, planar_threshold = 0.1, 
            max_iterations=50, tolerance=1e-6, 
            voxel_size=1,
            initial_transformation = None):  
        
        # hyperparam
        self.alpha = alpha # weight for plane
        self.max_corr_distance = max_corr_distance # maximum correspondence distance
        self.k = k # number of neighbors to determine normal
        self.planar_threshold = planar_threshold # threshold for planar regions

        self.src = torch.from_numpy(src).to(self.device, dtype=torch.float32) # N x 3
        self.tgt = torch.from_numpy(tgt).to(self.device, dtype=torch.float32) # M x 3
        self.src = self.downsample_with_open3d(self.src, voxel_size=voxel_size) # downsample source point cloud, N' x 3
        self.tgt = self.downsample_with_open3d(self.tgt, voxel_size=voxel_size) # downsample target point cloud, M' x 3

        self.src_np = self.src.cpu().numpy() # source point cloud numpy , N' x 3
        self.tgt_np = self.tgt.cpu().numpy() # target point cloud numpy , M' x 3
        self.tree = KDTree(self.tgt_np) # target point cloud KDTree

        self.compute_normals() # compute normals for target point cloud
        self.normal_candidate() # determine planar regions

        prev_error = float('inf')
        
        if initial_transformation is not None:
            # 초기 변환 설정
            transformation = torch.from_numpy(initial_transformation).to(self.device, dtype=torch.float32) # 4 x 4
            src_hom = torch.cat((self.src, torch.ones((self.src.shape[0],1), device=self.device, dtype=torch.float32)), dim=1) # N' x 4
            self.src = (transformation @ src_hom.T).T[:, :3] # N' x 3
            self.src_np = self.src.cpu().numpy() # N' x 3
        else:
            transformation = torch.eye(4, dtype=torch.float32, device=self.device) # 4 x 4
        
        iter = 0
        
        for iteration in range(max_iterations):
            self.nearest_neighbor() 
            T = self.optimization(self.planar_mask) # optimization , 4 x 4
            transformation = T @ transformation # update transformation, 4 x 4
            src_hom = torch.cat((self.src, torch.ones((self.src.shape[0],1), device=self.device, dtype=torch.float32)), dim=1) # N' x 4
            self.src = (T @ src_hom.T).T[:, :3] # update source point cloud, N' x 3
            self.src_np = self.src.cpu().numpy() # N' x 3
            mean_error = self.correspondence_distances.mean() # mean error, 1
            if torch.abs(torch.tensor(prev_error - mean_error)) < tolerance:
                print(f'Converged at iteration {iteration}')
                iter = iteration
                break
            prev_error = mean_error.item() # update previous error
        
        # Visualize normal candidates
        self.visualize_normal_candidates(normal_length = 3.0,mask=self.planar_mask)
        return transformation.cpu().numpy(), iter


    def optimization(self, mask):
        
        mask = mask.to(self.device) # M'
        mask_correpondence = mask[self.correspondence_indices] # N''

        # N'' = N''' + N'''
        plane_mask = (mask_correpondence == 1) # N'''
        point_mask = (mask_correpondence != 1) # N''''

        #cnt_pl = plane_mask.sum().item()
        #cnt_po = point_mask.sum().item()
        #print(f"Plane: {cnt_pl}, Point: {cnt_po}")

        src_plane = self.src_correspondence[plane_mask] # N''' 
        tgt_plane = self.tgt_correspondence[plane_mask] # N''' 
        normals_plane = self.normals[self.correspondence_indices[plane_mask]] # N''' 

        src_point = self.src_correspondence[point_mask] # N''''
        tgt_point = self.tgt_correspondence[point_mask] # N''''

        residual_plane, jac_plane = self.compute_plane_residual_jacobian(src_plane, tgt_plane, normals_plane) # N''' x 1, N''' x 1 x 6
        weight_plane = self.alpha

        residual_point, jac_point = self.compute_point_residual_jacobian(src_point, tgt_point) # N'''' x 3, N'''' x 3 x 6
        weight_point = 1 - self.alpha

        A = torch.zeros((6,6), device=self.device, dtype=torch.float32) # 6 x 6
        b = torch.zeros((6,1), device=self.device, dtype=torch.float32) # 6 x 1

        if residual_plane.numel() > 0:
            JT_plane = jac_plane.transpose(1,2) # N''' x 6 x 1
            JTR_plane = JT_plane @ residual_plane.unsqueeze(-1) # N''' x 6 x 1
            JTJ_plane = JT_plane @ jac_plane # N''' x 6 x 6
            A += weight_plane * JTJ_plane.sum(dim=0) # 6 x 6
            b += weight_plane * JTR_plane.sum(dim=0) # 6 x 1
       

        if residual_point.numel() > 0:
            JT_point = jac_point.transpose(1,2) # N'''' x 6 x 3
            JTR_point = torch.einsum('nij,nj->ni', JT_point, residual_point) # N'''' x 6
            JTR_point = JTR_point.unsqueeze(-1) # N'''' x 6 x 1
            JTJ_point = torch.einsum('nij,njk->nik', JT_point, jac_point) # N'''' x 6 x 6
            A += weight_point * JTJ_point.sum(dim=0) # 6 x 6
            b += weight_point * JTR_point.sum(dim=0) # 6 x 1
         

        solution = torch.linalg.lstsq(A, b) # 6 x 1
        delta = solution.solution # 6 x 1
        translation = delta[:3].flatten() # 3
        rotation = delta[3:].flatten() # 3
        R = rotation_matrix_from_rotvec(rotation) # 3 x 3

        T = torch.eye(4, dtype=torch.float32, device=self.device) # 4 x 4
        T[:3, :3] = R # 3 x 3
        T[:3, 3] = translation # 3 x 3
        return T

    def compute_plane_residual_jacobian(self, src_pts, tgt_pts, normals):
        '''
        residual r (1 x 1)
        r[n] = dot((tgt_pts[n] - src_pts[n]), normals[n])
        Jacobian matrix J (1 x 6)
        J[n] = [[ nx, ny, nz, (py * nz - pz * ny), (pz * nx - px * nz), (px * ny - py * nx) ]]
        Check https://arxiv.org/abs/2411.06766 (GenZ-ICP)
        '''
        # normal shape : N''' x 3
        diff = tgt_pts - src_pts # N''' x 3
        residual_val = torch.sum(diff * normals, dim=1) # N'''
        residual_val = residual_val.unsqueeze(1)  # N''' x 1

        cross_p = torch.stack([
            src_pts[:,1]*normals[:,2]-src_pts[:,2]*normals[:,1],
            src_pts[:,2]*normals[:,0]-src_pts[:,0]*normals[:,2],
            src_pts[:,0]*normals[:,1]-src_pts[:,1]*normals[:,0]
        ], dim=1) # N''' x 3

        jac = torch.zeros((src_pts.shape[0],1,6), dtype=torch.float32, device=self.device) # N''' x 1 x 6
        jac[:,0,0:3] = normals # N''' x 3
        jac[:,0,3:6] = cross_p # N''' x 3
        
        return residual_val, jac

    def compute_point_residual_jacobian(self, src_pts, tgt_pts):
        '''
        reisual r (1 x 3)
        r[n] = (r_x, r_y, r_z) = (tgt_x - src_x, tgt_y - src_y, tgt_z - src_z)
        Jacobian Matrix J (3 x 6)
        J[n] =  [ 1,  0,  0,  0,   z,  -y], 
                  [ 0,  1,  0,  -z,  0,   x],  
                  [ 0,  0,  1,   y, -x,   0]]  
        Check https://arxiv.org/abs/2411.06766 (GenZ-ICP)
        '''
        residual_val = tgt_pts - src_pts # N'''' x 3

        jac = torch.zeros((src_pts.shape[0],3,6), dtype=torch.float32, device=self.device) # N'''' x 3 x 6
        jac[:,0,0] = 1.0
        jac[:,1,1] = 1.0
        jac[:,2,2] = 1.0

        x = src_pts[:,0] # N''''
        y = src_pts[:,1] # N''''
        z = src_pts[:,2] # N''''

        jac[:,0,4] = z # N'''' x 1
        jac[:,0,5] = -y # N'''' x 1
        jac[:,1,3] = -z # N'''' x 1
        jac[:,1,5] = x # N'''' x 1
        jac[:,2,3] = y # N'''' x 1
        jac[:,2,4] = -x # N'''' x 1

        return residual_val, jac
    
    def visualize_normal_candidates(self, normal_length=3.0, mask=None):
        # Points and normals
        mask_correspondence = mask[self.correspondence_indices]
        points = self.tgt_correspondence.detach().cpu().numpy()
        normals = self.normals[self.correspondence_indices].detach().cpu().numpy()

        # Separate planar and non-planar points
        planar_points = points[mask_correspondence == 1]
        non_planar_points = points[mask_correspondence == 0]
        planar_normals = normals[mask_correspondence == 1]
        non_planar_normals = normals[mask_correspondence == 0]

        # Create point clouds
        planar_pcd = o3d.geometry.PointCloud()
        planar_pcd.points = o3d.utility.Vector3dVector(planar_points)
        planar_pcd.paint_uniform_color([1, 0, 0])  # Red for planar points

        non_planar_pcd = o3d.geometry.PointCloud()
        non_planar_pcd.points = o3d.utility.Vector3dVector(non_planar_points)
        non_planar_pcd.paint_uniform_color([0, 0, 1])  # Blue for non-planar points

        # Create lines for normals
        line_points = []
        line_colors = []
        line_indices = []
        index = 0

        for p, n in zip(planar_points, planar_normals):
            line_points.append(p)
            line_points.append(p + n * normal_length)
            line_indices.append([index, index + 1])
            line_colors.append([0, 1, 0])  # Green
            index += 2

        for p, n in zip(non_planar_points, non_planar_normals):
            line_points.append(p)
            line_points.append(p + n * normal_length)
            line_indices.append([index, index + 1])
            line_colors.append([0, 1, 0])  # Green
            index += 2

        line_points = np.array(line_points)
        line_indices = np.array(line_indices)
        line_colors = np.array(line_colors)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector(line_indices),
        )
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        # Visualize
        o3d.visualization.draw_geometries([planar_pcd, non_planar_pcd, line_set])

