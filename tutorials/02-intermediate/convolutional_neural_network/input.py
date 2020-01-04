import argparse, sys


def getargs():
    ## Construct an ArgumentParser object for command-line arguments
    parser = argparse.ArgumentParser(description='''Train CNN model with Hi-C data and make predictions''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('-i', '--input_path',
                     help='path to a .hic file.', type=str,
                        default='/data2/users/zengys/data/laten_train_data_0.1.npy')
    parser.add_argument('-o', '--target',
                     help='target of the .hic file',
                     type=str,
                     default='/data2/users/zengys/data/labels.mat')
    parser.add_argument('-t', '--type',
                        help='type of the .hic file',
                        type=str, default='odd')
    parser.add_argument('-u', '--upper_k',
                        help='target of the .hic file',
                        type=int, default=10)
    parser.add_argument('-d', '--down_k',
                        help='target of the .hic file',
                        type=int, default=10)

    commands = sys.argv[1:]
    if ((not commands) or ((commands[0] in ['inputfile'])
        and len(commands) == 1)):
        commands.append('-h')
    args = parser.parse_args(commands)

    return args, commands
