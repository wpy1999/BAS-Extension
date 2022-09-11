def get_hyperparameters(args, batch_flag=True):
    try:
        args.alpha = {'CUB_200_2011':{'vgg':0, 'resnet':0.5, 'inception':0.5, 'mobilenet':0.5},
                'ILSVRC':{'vgg':0.05, 'resnet':1, 'inception':1, 'mobilenet':0.5},
                'OpenImage':{'resnet':0.5}}[args.root][args.arch]
    except:  print('no alpha')
    try:
        args.beta = {'CUB_200_2011':{'vgg':0.9, 'resnet':1.4, 'inception':1, 'mobilenet':1.9},
            'ILSVRC':{'vgg':1, 'resnet':2.5, 'inception':2.5, 'mobilenet':1.5},
            'OpenImage':{'resnet':1.5}}[args.root][args.arch]   
    except:  print('no beta')
    try:
        args.topk = {'CUB_200_2011':{'vgg':80, 'resnet':200, 'inception':200, 'mobilenet':80},
            'ILSVRC':{'vgg':1, 'resnet':1, 'inception':1, 'mobilenet':1},
            'OpenImage':{'resnet':1}}[args.root][args.arch] 
    except:  print('no topk')
    try:
        args.threshold = {'CUB_200_2011':{'vgg':[0.1], 'resnet':[0.1], 'inception':[0.1], 'mobilenet':[0.1]},
            'ILSVRC':{'vgg':[0.45], 'resnet':[0.25], 'inception':[0.25], 'mobilenet':[0.5]}}[args.root][args.arch]
    except:  print('no threshold')
    try:
        args.epoch = {'CUB_200_2011': 120,
             'ILSVRC': 9,
             'OpenImage': 50}[args.root] 
    except:  print('no epoch')
    try:
        args.decay_epoch = {'CUB_200_2011': 80,
                  'ILSVRC': 3,
                  'OpenImage': 20}[args.root] 
    except:  print('no decay_epoch')
    try:
        if batch_flag:
            args.batch_size = {'CUB_200_2011':{'vgg':32, 'resnet':32, 'inception':32, 'mobilenet':32},
                    'ILSVRC':{'vgg':256, 'resnet':128, 'inception':128, 'mobilenet':128},
                    'OpenImage':{'resnet':64}}[args.root][args.arch]
    except:  print('no batch_size')

    return args