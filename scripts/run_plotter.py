import os
import pickle
import argparse
from collections import namedtuple

import pandas as pd
import geopandas as gpd

from utils.plotter import SWAIN_Plotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to experiment folder.')
    parser.add_argument("--exp-path", type=str, default='')
    parser.add_argument("--dataset", type=str, default='lamah')

    args = parser.parse_args()

    # Load objs
    obj_path = os.path.join(args.exp_path, 'eval_dump.pickle')
    with open(obj_path, 'rb') as file:
        results_dict = pickle.load(file)

    res_obj = namedtuple("Results", results_dict.keys())(*results_dict.values())

    if args.dataset == 'lamah':
        refs_df = pd.read_hdf('./data/LamaH-CE/lamah_refs.h5', 'df')

        rivernet = gpd.read_file('./data/LamaH-CE/E_stream_network/RiverATLAS.shp')
        basins = gpd.read_file('./data/LamaH-CE/B_basins_intermediate_all/3_shapefiles/Basins_B.shp')

        rivernet = rivernet[rivernet['ORD_STRA'] > 1]
        basins = basins[basins.ID.isin(res_obj.node_idx_map.keys())]

        rivernet = rivernet[['geometry']].dropna()
        basins = basins[['geometry']].dropna()

        info_map_shp = [dict(source=rivernet, color='deepskyblue', line_width=1)]
                        # dict(source=basins, color='yellow', line_width=1)]
        default_node = 399
        crs = 'EPSG:3035'
    else:
        refs_df = pd.DataFrame()
        info_map_shp = None
        default_node = 80
        crs = 'EPSG:3857'

    plotter = SWAIN_Plotter(results=res_obj,
                            def_node=default_node,
                            reference_df=refs_df,
                            crs=crs,
                            info_map_shp=info_map_shp)
    server = plotter.run()

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()