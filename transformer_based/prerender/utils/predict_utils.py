import argparse
import yaml
from yaml import Loader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=False, help="Model to load")
    parser.add_argument("--model-name-addition", type=str, required=False, help="Addition for savefolder")
    parser.add_argument("--test-data-path", type=str, required=False, help="Path to prerendered training data")
    parser.add_argument("--out-path", type=str, required=False, help="Path to predictions folder")
    parser.add_argument("--viz-path", type=str, required=False, help="Path to visualization folder")
    parser.add_argument("--n-jobs", type=int, required=False, help="Dataloader number of workers")
    parser.add_argument("--batch-size", type=int, required=False, help="Dataloader batch size")
    parser.add_argument("--n-shards", type=int, required=False,
                        help="n_shards for test data. 1/n_shards will be used")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    # ToDo: make all config parameters overridable from command line. Remember to update get_config method

    args = parser.parse_args()
    return args


def get_config(args):
    path = args.config
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader)

    # If provided in command line, override defaults
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.model_name_addition is not None:
        config["model"]["name_addition"] = args.model_name_addition

    if args.test_data_path is not None:
        config["test"]["data_config"]["dataset_config"]["data_path"] = args.test_data_path

    if args.out_path is not None:
        config["test"]["output_config"]["out_path"] = args.out_path

    if args.viz_path is not None:
        config["test"]["output_config"]["viz_path"] = args.viz_path

    if args.n_jobs is not None:
        config["test"]["data_config"]["dataloader_config"]["num_workers"] = args.n_jobs

    if args.batch_size is not None:
        config["test"]["data_config"]["dataloader_config"]["batch_size"] = args.batch_size

    if args.n_shards is not None:
        config["test"]["data_config"]["dataset_config"]["n_shards"] = args.n_shards

    return config