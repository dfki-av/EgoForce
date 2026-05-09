import os
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS

from .Arctic import Arctic
from .COCOWholeBody import COCOWholeBody


@DATASETS.register_module()
class CombinedDataset(BaseDetDataset):
    METAINFO = {
       'classes': ('left_forearm', 'right_forearm', 'left_hand', 'right_hand'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }


    def load_data_list(self):
        args = []
        kwargs = {}

        subset = self.data_prefix['subset'].split(os.path.sep)[-1]
        kwargs['data_prefix'] = {'subset': subset}
        
        kwargs['lazy_init'] = True

        kwargs['data_root']  = '/fscratch/millerdurai/Datasets/arctic/ArcticSeqNoCropImages'
        # kwargs['ann_file']  = '/fscratch/millerdurai/Datasets/arctic/arm_annotations.json'
        kwargs['ann_file']  = '/fscratch/millerdurai/Datasets/arctic/arm_annotations_with_hands.json'

        arctic_dataset = Arctic(*args, **kwargs)
 
        kwargs['data_root']  = '/netscratch/millerdurai/Datasets/COCO_KPTS/2017/COCO_WholeBody'
        # kwargs['ann_file']  = f'/netscratch/millerdurai/Datasets/COCO_KPTS/2017/{subset}_arm_annotations.json'
        kwargs['ann_file']  = f'/netscratch/millerdurai/Datasets/COCO_KPTS/2017/{subset}_arm_annotations_with_hands.json'

        coco_wholebody_dataset = COCOWholeBody(*args, **kwargs)
 

        print("Loading data list for CombinedDataset")
        data_list = []
        for dataset in [arctic_dataset, coco_wholebody_dataset]:
            data_list += dataset.load_data_list()

        return data_list
