import numpy as np

# Modify later to support getting the parameters for either camera, since the dataset will come from both

# Camera 1 parameters (attached to the dark grey mount, Luke uses)
class OrbbecCamParams:
    def __init__(self, cam_num, im_dim=(640,480)):
        if cam_num not in [1,2]:
            ValueError('Invalid camera selected. Please choose 1 (Luke\'s) or 2 (Matteo\'s).')
        self.w=640
        self.h=480
        self.scale_factor = 1.0

        if im_dim != (640,480):
            self.w = im_dim[0]
            self.h = im_dim[1]
            self.scale_factor = self.w/640.0 # 0.5 for 320x240

        if cam_num == 1:
            # Luke's camera

            # Depth is left camera
            self.fx_d = self.scale_factor*578.938
            self.fy_d = self.scale_factor*578.938
            self.cx_d = self.scale_factor*318.496
            self.cy_d = self.scale_factor*251.533

            self.k1_d = -0.094342
            self.k2_d = 0.290512
            self.p1_d = -0.299526
            self.p2_d = -0.000318
            self.k3_d = -0.000279

            self.cam_matrix_d = np.array([
                [self.fx_d,0,self.cx_d,0],
                [0,self.fy_d,self.cy_d,0],
                [0,0,1,0],
                [0,0,0,1]
            ])

            self.dist_d = np.array([ self.k1_d, self.k2_d, self.p1_d, self.p2_d, self.k3_d])

            # RGB is right camera
            self.fx_rgb = self.scale_factor*517.138
            self.fy_rgb = self.scale_factor*517.138
            self.cx_rgb = self.scale_factor*319.184
            self.cy_rgb = self.scale_factor*229.077

            self.k1_rgb = 0.044356
            self.k2_rgb = -0.174023
            self.p1_rgb = 0.077324
            self.p2_rgb = 0.001794
            self.k3_rgb = -0.003853

            self.cam_matrix_rgb = np.array([
                [self.fx_rgb,0,self.cx_rgb,0],
                [0,self.fy_rgb,self.cy_rgb,0],
                [0,0,1,0],
                [0, 0, 0, 1]
            ])

            self.dist_rgb = np.array([self.k1_rgb, self.k2_rgb, self.p1_rgb, self.p2_rgb,self.k3_rgb ])

            self.R_rgb_2_depth = np.array([
                [0.999992, -0.00334205 , 0.0022538, -25.0971],
                [0.00332987, 0.99998,0.00539669, 0.287867],
                [-0.00227178, -0.00538911, 0.999983, -1.11816],
                [0, 0, 0, 1]
            ])

            # Orbbec calls this "All_Mat", so I'll leave that notation
            # Alignment matrix? Not sure...
            temp = np.empty(self.R_rgb_2_depth.shape)
            self.All_Mat = np.empty(self.R_rgb_2_depth.shape)
            np.matmul(self.R_rgb_2_depth,np.linalg.inv(self.cam_matrix_d),temp)
            np.matmul(self.cam_matrix_rgb, temp, self.All_Mat)

            # Hard-coded from camera flash (from orbbec tool) (for 640x480 image)
            # self.All_Mat = np.array(([0.891994, -0.00595647, 37.7456, -13335.6],
            #                          [0.00207551, 0.891104, 7.06065, -107.278],
            #                          [-0.00000392406, -0.00000930862, 1.00357, -1.11816],
            #                          [0,0,0,1]))

            # print(self.All_Mat)
            self.mat = np.empty(16)
            self.mat[0] = self.All_Mat[0,0]
            self.mat[1] = self.All_Mat[0,1]
            self.mat[2] = self.All_Mat[0,2]
            self.mat[3] = self.All_Mat[0,3]
            self.mat[4] = self.All_Mat[1,0]
            self.mat[5] = self.All_Mat[1,1]
            self.mat[6] = self.All_Mat[1,2]
            self.mat[7] = self.All_Mat[1,3]
            self.mat[8] = self.All_Mat[2,0]
            self.mat[9] = self.All_Mat[2,1]
            self.mat[10] = self.All_Mat[2,2]
            self.mat[11] = self.All_Mat[2,3]
            self.mat[12] = self.All_Mat[3,0]
            self.mat[13] = self.All_Mat[3,1]
            self.mat[14] = self.All_Mat[3,2]
            self.mat[15] = self.All_Mat[3,3]


        if cam_num == 2:
            # Matteo's camera
            ValueError('Camera Parameters for cam 2 haven\'t been defined yet')