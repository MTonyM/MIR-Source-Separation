import argparse       


def get_args():
    def check_args(args):
        if args.train:
            assert args.train_manifest is not None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load weights')
    parser.add_argument('--batch_size', help='batch size', default=16)
    parser.add_argument('--max_epoch', help='load model', default=80)
    parser.add_argument('--num_epoches', help='epoches num', default=5)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-3)
    parser.add_argument('--len_block',help='block size',default=1024)
    parser.add_argument('--len_frame',help='len_frame',default=1024)
    parser.add_argument('--len_hop',help='len_hop',default=512)
    parser.add_argument('--log_dir', help="directory of logging", default=None) 
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_manifest',  help='train_textfile',default='data_train.txt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--sample_rate', help='sample rate', default=16000)
    parser.add_argument('--name', help='comma separated list of GPU(s) to use.', default = 'data')
    parser.add_argument('--test_ratio', help='ratio of test data', default = 0.1)
    parser.add_argument('--dir', help="directory of logging", default='/home/maoym/MIR-separation/dataset/iKala/Wavfile')

    args = parser.parse_args()
    check_args(args)
    return args        