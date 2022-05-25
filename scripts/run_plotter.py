import os
import sys
import pickle
import argparse

# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

from utils.plotter import SWAIN_Plotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to experiment folder.')
    parser.add_argument("--exp-path", type=str, default='')

    args = parser.parse_args()

    # Load obj
    obj_path = os.path.join(known_args.eval_path, 'eval_dump.pickle')
    with open(obj_path, 'rb') as file:
        results_dict = pickle.load(file)

    results = namedtuple("Results", results_dict.keys())(*results_dict.values())

    plotter = SWAIN_Plotter(results)

    plotter.build()
    server = plotter.run()

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()