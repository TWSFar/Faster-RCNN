class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return '/home/twsf/work/Faster-RCNN/data/VOC2012'
        elif dataset == 'visdrone':
            return '/home/twsf/work/Faster-RCNN/data/VisDrone'
        elif dataset == 'hkb':
            return '/home/twsf/work/Faster-RCNN/data/HKB'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return 'G:\\CV\\weights\\resnet101.pth'
        elif backbone == 'vgg16':
            return 'G:\\CV\\weights\\vgg16.pth'
        else:
            print('weights {} not available.'.format(backbone))
            raise NotImplementedError
