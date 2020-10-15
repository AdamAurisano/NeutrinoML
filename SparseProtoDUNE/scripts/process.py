'''
Preprocess ROOT-format sparse pixel maps into PyTorch format
'''

import yaml, argparse, logging
from SparseProtoDUNE import datasets

def parse_args():
    '''Parse arguments'''
    parser = argparse.ArgumentParser('process.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='/scratch/SparseProtoDUNE/config/sparse_3d.yaml')
    return parser.parse_args()

def configure(config):
    '''Load configuration'''
    with open(config) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def main():
    '''Main function'''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initialising')
    args = parse_args()
    config = configure(args.config)
    logging.info(f'Loading {config["data"]["name"]} dataset from {config["data"]["root"]}')
    dataset = datasets.get_dataset(**config['data'])
    dataset.process(**config['process'])

if __name__ == '__main__':
    main()

