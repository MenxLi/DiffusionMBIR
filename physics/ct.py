import torch
import numpy as np
from .radon import Radon, IRadon
from .radon.radon3d import Radon3D, IRadon3D
from abc import ABC, abstractmethod

class CT_abstract(ABC):
    @abstractmethod
    def A(self, x):...
    @abstractmethod
    def A_dagger(self, y):...
    @abstractmethod
    def AT(self, y):...

class CBCT(CT_abstract):
    def __init__(self, radon_view, geo, half_circle = True, device='cuda:0'):
        if half_circle:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
            theta_all = np.linspace(0, 180, 180, endpoint=False)
        else:
            raise NotImplementedError

        self.radon = Radon3D(geo, theta).to(device)
        self.radon_all = Radon3D(geo, theta_all).to(device)
        self.iradon_all = IRadon3D(geo, theta_all).to(device)
        self.iradon = IRadon3D(geo, theta).to(device)
        self.radont = IRadon3D(geo, theta, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_all(self, x):
        return self.radon_all(x)

    def A_all_dagger(self, x):
        return self.iradon_all(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)

class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
            theta_all = np.linspace(0, 180, 180, endpoint=False)
        else:
            theta = torch.arange(radon_view)
            theta_all = torch.arange(radon_view)

        self.radon = Radon(img_width, theta, circle).to(device)
        self.radon_all = Radon(img_width, theta_all, circle).to(device)
        self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_all(self, x):
        return self.radon_all(x)

    def A_all_dagger(self, x):
        return self.iradon_all(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)


class CT_LA(CT_abstract):
    """
    Limited Angle tomography
    """
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)
