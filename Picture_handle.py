from PIL import  Image
import struct
import numpy as np
import matplotlib.pyplot as mlt
import PCA as pca


class picture_handle:
    def __init__(self,aim_dim):
        self.src_dim = 28
        self.aim_dim = aim_dim
        self.src_dir = "t10k-images-idx3-ubyte"
        with open(self.src_dir, 'rb') as f:
            data = f.read(16)
            des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
            train_x = np.zeros((img_nums, row * col))
            for index in range(img_nums):
                data = f.read(784)
                if len(data) == 784:
                    train_x[index, :] = np.array(struct.unpack_from('>' + 'B' * (row * col), data, 0)).reshape(1, 784)
            f.close()
            self.x_matrix = train_x[0].reshape(28,28)
            print(self.x_matrix)



ph = picture_handle(3)
mlt.imshow(ph.x_matrix)
mlt.show()
temp_pca = pca.PCA(exp_dim=1,src_dim=28,x_matrix=ph.x_matrix,sum=28)
temp_pca.main_fuction()
mlt.imshow(temp_pca.revert_matrix)
mlt.show()

