import argparse
import yaml
from helper import Helper


from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIA')
    parser.add_argument('--params', dest='params', default='configs/default_params.yaml')

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    helper = Helper(params)
    helper.test_attack()