class Params_ResnetFilter_Img(object):
    def __init__(self, granularity):
        # inp, oup, exp
        IMG_GRAN_LOW = [
            [16, 32, 64, 128],
            [8, 16, 32, 64],
        ]
        IMG_GRAN_MED = [
            [16, 16, 16, 32],
            [8, 8, 8, 16],
        ]

        if granularity=='low':
            self.granularity = IMG_GRAN_LOW
            self.gates_fixed_open=1
        elif granularity=='medium':
            self.granularity = IMG_GRAN_MED
            self.gates_fixed_open=2
        else:
            assert(False)




class Params_ResnetFilter_Cifar(object):
    def __init__(self, granularity):
        # inp, oup, exp
        CIFAR_GRAN_LOW = [8, 8, 8]
        CIFAR_GRAN_MEDIUM = [4, 4, 4]
        CIFAR_GRAN_HIGH = [2, 2, 2]
        CIFAR_GRAN_SUPERHIGH = [1, 1, 1]

        if granularity=='low':
            self.granularity = CIFAR_GRAN_LOW
            self.gates_fixed_open=0
        elif granularity=='medium':
            self.granularity = CIFAR_GRAN_MEDIUM
            self.gates_fixed_open=0
        elif granularity=='high':
            self.granularity = CIFAR_GRAN_HIGH
            self.gates_fixed_open=0
        elif granularity=='superhigh':
            self.granularity = CIFAR_GRAN_SUPERHIGH
            self.gates_fixed_open=0
        else:
            assert(False)



