# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# Changed by Xinchen Liu

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17

from .aic import AICity19
from .veri import VeRi
from .vehicleid import VehicleID
from .veriwild import VeRiWild
from .vehicleonem import VehicleOneM
from .vd1 import VD1
from .vd2 import VD2

from .veri_mask import VeRi_Mask
from .veriwild_mask import VeRiWild_Mask
from .veriwild_small import VeRiWild_Small
from .veriwild_small_mask import VeRiWild_Small_Mask
from .veriwild_medium import VeRiWild_Medium
from .veriwild_medium_mask import VeRiWild_Medium_Mask

from .vehicleid_mask import VehicleID_Mask
from .vehicleid_small import VehicleID_Small
from .vehicleid_small_mask import VehicleID_Small_Mask


from .dataset_loader import *

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    
    'aic': AICity19,
    'vehicleid': VehicleID,
    'veri': VeRi,
    'veriwild': VeRiWild,
    'vehicleonem': VehicleOneM,
    'vd1': VD1,    
    'vd2': VD2, 

    'veri_mask': VeRi_Mask,
    'veriwild_mask': VeRiWild_Mask,
    
    'veriwild_small': VeRiWild_Small,
    'veriwild_small_mask': VeRiWild_Small_Mask,
    'veriwild_medium': VeRiWild_Medium,
    'veriwild_medium_mask': VeRiWild_Medium_Mask,
    
    'vehicleid_mask': VehicleID_Mask,
    'vehicleid_small': VehicleID_Small,
    'vehicleid_small_mask': VehicleID_Small_Mask
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
