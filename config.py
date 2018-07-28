import argparse
import os


def get_args():
    def check_args(args):
        if args.train:
            assert args.train_manifest is not None

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use.', default='rnn')
    parser.add_argument('--dataset', help='dataset to use.', default='iKala_aug')
    parser.add_argument('--load', help='load weights')
    parser.add_argument('--augment', help='if do augmentation', default=True, type=str2bool)
    parser.add_argument('--debug', help='if debug', default=False, type=str2bool)
    parser.add_argument('--batch_size', help='batch size', default=64)
    ######################### train par
    parser.add_argument('--num_epoches', help='epoches num', default=5)
    parser.add_argument('--epochNum', default=-1, type=int, help='0=retrain | -1=latest | -2=best', choices=[0, -1, -2])
    parser.add_argument('--learning_rate', help='learning rate', default=1e-3)
    parser.add_argument('--aug', help='wheter do data augmentation', default=True)

    ######################### audio info/par
    parser.add_argument('--len_frame', help='len_frame', default=1024)
    parser.add_argument('--len_hop', help='len_hop', default=512)
    parser.add_argument('--sample_rate', help='sample rate', default=16000, type=int)

    parser.add_argument('--log_dir', help="directory of logging", default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--name', help='comma separated list of GPU(s) to use.', default='data')
    parser.add_argument('--dir', help="directory of logging",
                        default='/home/maoym/MIR-separation/dataset/iKala/Wavfile')
    #     parser.add_argument('--dir', help="directory of logging", default='/home/maoym/MIR-separation/dataset/Wavfile/train')
    parser.add_argument('--GPUs', help='comma separated list of GPU(s) to use.', default='1', type=str)
    parser.add_argument('--GPU', help='if use gpu.', default=True, type=str2bool)
    parser.add_argument('--resume', default='../models', type=str, help='Path to checkpoint')

    ########### train \ crossValidation \ test

    parser.add_argument('--mode', help='0: train 1:test', default=0)

    parser.add_argument('--train_ration', help='ration of trainset in dataset', default=0.9)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUs

    args.hashKey = args.dataset + '_' + args.model
    args.resume = os.path.join(args.resume, args.hashKey)
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    check_args(args)
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
