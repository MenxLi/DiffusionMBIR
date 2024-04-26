
import torch
import astra
import numpy as np
from typing import Tuple, Sequence, Optional, TypedDict

def geoGetter(
    camera_distance: float,
    im_shape_HW: Tuple[int, int],
    pixel_size: float,
    focal_len: int,
    volume_shape: Tuple[int, int, int],
    voxel_size: float = 1.0,
):
    pixel_size_original = np.array((pixel_size, pixel_size))
    pixel_size_calc = (camera_distance * 2 * pixel_size_original) / focal_len
    return {
        'det_spacing_x': pixel_size_calc[1] * voxel_size,
        'det_spacing_y': pixel_size_calc[0] * voxel_size,
        'det_row_count': im_shape_HW[0] * voxel_size,
        'det_col_count': im_shape_HW[1] * voxel_size,
        'source_origin': camera_distance * voxel_size,
        'origin_det': camera_distance * voxel_size,
        'vol_shape': volume_shape,
    }

class _ProjcectionConfig(TypedDict):
    camera_distance: float
    im_shape_HW: Tuple[int, int]
    pixel_size: float
    focal_len: int
    volume_shape: Tuple[int, int, int]
    voxel_size: float = 1.0
class Radon3D:
    """
    Projection with Tigre toolbox
    """
    def __init__(self, geo: _ProjcectionConfig, angles: Sequence[float]):
        # input angles in degrees!
        self.angles = angles * np.pi / 180
        self.geo = geoGetter(**geo)
        self.device = 'cuda'
    
    def __call__(self, volume: torch.Tensor):
        """
        # - volume: volume of shape (Z, ., .)
        # (may be add a batch dimension to the volume, (B, Z, ., .)?)
        # return:
        #     proj: projection of shape (angles, detector_h, detector_w)

        - volume: volume of shape (Z, 1, ., .)
        return:
            proj: projection of shape (detector_h, 1, detector_w, angles)
        """
        # print("Radon")
        assert len(volume.shape) == 4 and volume.shape[1] == 1
        volume = volume.squeeze(1)

        volume = volume.detach().cpu().numpy()

        assert self.geo['vol_shape'] == volume.shape, f"volume.shape: {volume.shape}, self.geo[\'vol_shape\']: {self.geo['vol_shape']}"
        vol_geom = astra.create_vol_geom(volume.shape[2], volume.shape[1], volume.shape[0])

        proj_geom = astra.create_proj_geom(
            'cone', 
            self.geo['det_spacing_x'], self.geo['det_spacing_y'],
            self.geo['det_row_count'], self.geo['det_col_count'],
            self.angles,
            self.geo['source_origin'], self.geo['origin_det']
            )

        # do the projection
        sino_id, sino = astra.create_sino3d_gpu(volume, proj_geom, vol_geom)
        astra.data3d.delete(sino_id)

        # projections of shape (angles, detector_h, detector_w)
        projections = torch.tensor(sino, device = self.device).permute(1, 0, 2)

        return projections.permute(1, 2, 0).unsqueeze(1)
    
    def to(self, device):
        self.device = device
        return self

class IRadon3D:
    """
    Inverse Projection with Tigre toolbox (FDK)
    """
    def __init__(
        self, geo, 
        angles: Sequence[float], 
        use_filter: Optional[bool] = True,
        ):

        # input angles in degrees!
        self.angles = angles * np.pi / 180
        self.use_filter = use_filter
        self.geo = geoGetter(**geo)
        self.out_size = self.geo['vol_shape']
        self.device = 'cuda'
    
    def to(self, device):
        self.device = device
        return self
    
    def __call__(self, projs: torch.Tensor) -> torch.Tensor:
        """
        # - projs: projection of shape (angles, detector_h, detector_w)
        # return:
        #     volume: volume of shape (Z, ., .)

        - projs: projection of shape (detector_h, 1, detector_w, angles)
        return:
            volume: volume of shape (Z, 1, ., .)
        """
        # print("IRadon")
        assert len(projs.shape) == 4 and projs.shape[1] == 1
        projs = projs.permute(1, 3, 0, 2).squeeze(0)

        # projs = projs.permute(2, 0, 1).detach().cpu().numpy()
        projs = projs.permute(1, 0, 2).detach().cpu().numpy()

        proj_geom = astra.create_proj_geom(
            'cone', 
            self.geo['det_spacing_x'], self.geo['det_spacing_y'],
            self.geo['det_row_count'], self.geo['det_col_count'],
            self.angles,
            self.geo['source_origin'], self.geo['origin_det']
            )
        vol_geom = astra.create_vol_geom(self.out_size[2], self.out_size[1], self.out_size[0])
        # (Z, ., .)
        if not self.use_filter:
            # vol_id, volume = astra.create_backprojection3d_gpu(projs, proj_geom, vol_geom, gpuIndex=1)
            # astra.data3d.delete(vol_id)
            sino_id = astra.data3d.create('-sino', proj_geom, projs)
            vol_id = astra.data3d.create('-vol', vol_geom, 0)
            cfg = astra.astra_dict('BP3D_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = vol_id
            # cfg['option']={'GPUindex':1}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            volume = astra.data3d.get(vol_id)

            astra.data3d.delete(sino_id)
            astra.data3d.delete(vol_id)
            astra.algorithm.delete(alg_id)

            volume = volume * np.pi / (2 * len(self.angles))
        else:
            # use FDK
            sino_id = astra.data3d.create('-sino', proj_geom, projs)
            vol_id = astra.data3d.create('-vol', vol_geom, 0)
            cfg = astra.astra_dict('FDK_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = vol_id
            # cfg['option']={'GPUindex':1}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            volume = astra.data3d.get(vol_id)

            astra.algorithm.delete(alg_id)
            astra.data3d.delete(sino_id)
            astra.data3d.delete(vol_id)
        ret = torch.tensor(volume, device=self.device, requires_grad=False)
        # print("IRadon Done")
        return ret.unsqueeze(1)
