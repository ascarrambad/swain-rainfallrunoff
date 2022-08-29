import os
import pickle
import math
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

from tsl import logger
from tsl.utils import download_url
from tsl.utils.python_utils import files_exist
from tsl.ops.similarities import gaussian_kernel
from tsl.ops.connectivity import adj_to_edge_index
from tsl.datasets.prototypes import PandasDataset
from tsl.data.datamodule.splitters import AtTimeStepSplitter

from swaindb.parse_ergene import get_data


class Ergene(PandasDataset):
    url = None

    similarity_options = {'distance', 'stcn', 'binary'}
    temporal_aggregation_options = None
    spatial_aggregation_options = None

    def __init__(self,
                 root='./data/',
                 freq='1D',
                 \
                 replace_nans=True,
                 mask_u=False,
                 \
                 selected_ids=None,
                 k_hops=None,
                 ):
        # Set root path
        self.root = root

        # load dataset
        ts_qobs_df, ts_qobs_mask, ts_exos_dict, attribs_dict, binary_mtx = self.load(replace_nans=replace_nans,
                                                                                     mask_u=mask_u,
                                                                                     selected_ids=selected_ids,
                                                                                     k_hops=k_hops)

        super().__init__(dataframe=ts_qobs_df,
                         mask=ts_qobs_mask,
                         exogenous=ts_exos_dict,
                         attributes=attribs_dict,
                         freq=freq,
                         similarity_score="binary",
                         name="Ergene")

        self.distance_connectivity = None
        self.binary_connectivity = binary_mtx
        self.test_start = datetime(2005,1,1)

    @property
    def raw_files_paths(self):
        return ['./data/Ergene']

    @property
    def required_file_names(self):
        return ['Ergene/ergene_qobs.csv',
                'Ergene/ergene_exos.pickle',
                'Ergene/ergene_attribs.pickle',
                'Ergene/ergene_hierarchy.npy']

    def build(self) -> None:
        logger.info('Building data...')

        # Extract data from db and create folders
        ts_qobs_df, ts_exos_df, _, gdf_locations = get_data()

        directory = './data/Ergene'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Clean weather data and match with catchments
        weather_node_ids = [87, 80, 76, 85, 91, 92, 82, 94, 77, 96, 83, 84, 99, 100]
        w_node_rem = [87, 91, 92, 94, 96, 99, 100]
        w_node_add = [78, 79, 81, 86]
        flow_node_ids = [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]

        ts_exos_df.columns.set_levels(weather_node_ids, level=0, inplace=True)
        ts_exos_df = ts_exos_df.drop(w_node_rem, axis=1, level=0)
        ts_exos_df = ts_exos_df.join(pd.DataFrame(np.nan,
                                                  columns=pd.MultiIndex.from_product([w_node_add, [258, 259, 260]]),
                                                  index=ts_exos_df.index))

        ts_qobs_df.index = ts_qobs_df.index.normalize()
        ts_exos_df.index = ts_exos_df.index.normalize()

        ts_exos_df = ts_exos_df.drop(ts_exos_df.tail(1).index)
        ts_qobs_df = ts_qobs_df.loc[ts_exos_df.index[0]:]
        ts_qobs_df = ts_qobs_df.loc[:ts_exos_df.index[-1]]
        ts_exos_df = ts_exos_df.reindex(index=ts_qobs_df.index)

        ts_qobs_df = ts_qobs_df.sort_index(axis=1)
        ts_exos_df = ts_exos_df.sort_index(axis=1, level=0)

        ts_exos_dict = {'u': ts_exos_df}

        # Build distances matrix
        keep_ids = list(ts_qobs_df.columns)
        hierarchy_mtx, edge_attr_df = self.build_hierarchy_matrix(keep_ids, gdf_locations['geometry'])

        # Build static attributes data
        catch_attr_df = gdf_locations
        catch_attr_df = catch_attr_df.filter(flow_node_ids, axis=0).sort_index(axis=0)
        catch_attr_df['lat'] = catch_attr_df.geometry.x
        catch_attr_df['lon'] = catch_attr_df.geometry.y
        catch_attr_df = catch_attr_df[['lat', 'lon']]

        attribs_dict = {'catchment': catch_attr_df,
                        'stream': edge_attr_df}

        # Store built data
        ts_qobs_df.to_csv(os.path.join(self.root_dir, 'Ergene/ergene_qobs.csv'))

        with open(os.path.join(self.root_dir, 'Ergene/ergene_exos.pickle'), 'wb') as file:
            pickle.dump(ts_exos_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.root_dir, 'Ergene/ergene_attribs.pickle'), 'wb') as file:
            pickle.dump(attribs_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(os.path.join(self.root_dir, 'Ergene/ergene_hierarchy.npy'), hierarchy_mtx)

    def load_raw(self):
        self.maybe_build()

        # Data paths
        gauges_path = os.path.join(self.root_dir, 'Ergene/ergene_qobs.csv')
        exos_path = os.path.join(self.root_dir, 'Ergene/ergene_exos.pickle')
        attribs_path = os.path.join(self.root_dir, 'Ergene/ergene_attribs.pickle')
        binary_mtx_path = os.path.join(self.root_dir, 'Ergene/ergene_hierarchy.npy')

        # Load time-series data
        ts_qobs_df = pd.read_csv(gauges_path)
        ts_qobs_df['datetime'] = ts_qobs_df['datetime'].astype('datetime64[ns]')
        ts_qobs_df = ts_qobs_df.set_index('datetime').astype(np.float32, errors='ignore')
        ts_qobs_df.columns = ts_qobs_df.columns.map(int)

        with open(exos_path, 'rb') as file:
            ts_exos_dict = pickle.load(file)

        # Load attributes
        with open(attribs_path, 'rb') as file:
            attribs_dict = pickle.load(file)

        # Load distance matrix
        binary_mtx = np.load(binary_mtx_path)

        return ts_qobs_df, ts_exos_dict, attribs_dict, binary_mtx

    def load(self, replace_nans=True, mask_u=False, selected_ids=None, k_hops=None):
        data = self.load_raw()

        # Select catchment ids and k_hops
        ts_qobs_df, ts_exos_dict, attribs_dict, binary_mtx = self._select(*data,
                                                                          selected_ids=selected_ids,
                                                                          k_hops=k_hops)

        self.idx_node_map = {i: int(n) for i, n in enumerate(ts_qobs_df.columns)}
        self.node_idx_map = {n:i for i, n in enumerate(ts_qobs_df.columns)}

        # Transform eventual pd.DataFrames into ndarrays
        attribs_dict.update({k: df.to_numpy() for k, df in attribs_dict.items() if isinstance(df, pd.DataFrame)})

        # Compute validity mask and fill NaNs
        ts_qobs_df = ts_qobs_df.replace({-999: np.NaN})
        ts_qobs_mask = ~np.isnan(ts_qobs_df.values)
        if replace_nans:
            ts_qobs_df = ts_qobs_df.fillna(value=-1, axis=1)

        ts_exos_dict['u'].replace({-999: np.NaN}, inplace=True)
        ts_exos_mask = ~np.isnan(ts_exos_dict['u'].values.reshape(ts_qobs_mask.shape[0], len(ts_qobs_df.columns), -1))
        if replace_nans:
            ts_exos_dict['u'].fillna(value=-1, axis=1, inplace=True)

        if mask_u:
            ts_exos_dict['umask'] = ts_exos_mask

        return ts_qobs_df, ts_qobs_mask, ts_exos_dict, attribs_dict, binary_mtx

    def load_plotting_data(self):
        node_idx_map = {n:i for i, n in enumerate(self.nodes)}

        _, _, attribs_dict, _ = self.load_raw()

        return attribs_dict['catchment'], node_idx_map

    ############################################################################

    def build_hierarchy_matrix(self, ids, loc_srs):
        logger.info('Building hierarchy matrix...')

        # Build initial objects
        num_sensors = len(ids)
        hierarchy_edges = [[81,86], [79,86], [86,77], [77,76], [78,76], [84,76], [76,85], [83,85], [82,85], [85, 80]]
        hierarchy = np.zeros((num_sensors, num_sensors), dtype=np.int)

        # Builds node id to index map
        nim = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

        # Compute distance, elev_diff, slope and fill hierarchy matrix
        dist_hdns = []
        elev_diffs = []
        strm_slopes = []
        for e in hierarchy_edges:
            hierarchy[nim[e[0]],nim[e[1]]] = 1
            dist = loc_srs.loc[e[0]].distance(loc_srs.loc[e[1]])
            elevd = abs(loc_srs.loc[e[0]].z - loc_srs.loc[e[1]].z)
            slope = math.sqrt(dist**2 + elevd**2)

            dist_hdns.append(dist)
            elev_diffs.append(elevd)
            strm_slopes.append(slope)

        logger.info('Building edge features...')

        edge_attr_df = pd.DataFrame(dict(dist_hdn=dist_hdns,
                                         elev_diff=elev_diffs,
                                         strm_slope=strm_slopes),
                                    index=[e[0] for e in hierarchy_edges])
        edge_attr_df = edge_attr_df.sort_index(axis=0)

        return hierarchy, edge_attr_df

    ############################################################################

    def _select(self, ts_qobs_df, ts_exos_dict, attribs_dict, binary_mtx,
                selected_ids, k_hops=None):
        if selected_ids is None: return ts_qobs_df, ts_exos_dict, attribs_dict, binary_mtx

        # Create node to idx mapping
        idx_node_map = {i: int(n) for i, n in enumerate(ts_qobs_df.columns)}
        node_idx_map = {n:i for i, n in enumerate(ts_qobs_df.columns)}

        # Create indexes
        nodes_idxs = list(idx_node_map.keys())
        edge_index, _ = adj_to_edge_index(binary_mtx)
        edge_index = np.array(list(reversed(edge_index.tolist())))
        keep_idxs = list(map(node_idx_map.get, selected_ids))

        # Create graph
        g = nx.DiGraph()
        g.add_nodes_from(nodes_idxs)
        g.add_edges_from(edge_index.T)

        # Select k_oops
        if k_hops is not None:
            # Select k-hop-subgraph
            subgraphs = []
            for i in keep_idxs:
                k_hop_subgraph = nx.ego_graph(G=g,
                                              n=i,
                                              radius=k_hops)
                subgraphs.append(k_hop_subgraph.nodes)

            # Return for selection
            keep_idxs = sorted(list(set.union(*map(set, subgraphs))))
            selected_ids = list(map(idx_node_map.get, keep_idxs))

        g = g.subgraph(nodes=keep_idxs)

        # Select data
        ts_qobs_df = ts_qobs_df[selected_ids]

        for k, df in ts_exos_dict.items():
            ts_exos_dict[k] = df.iloc[:, df.columns.isin(selected_ids, level=0)]

        # Filtering out nodes with no outgoing edges
        keep_edges = [idx for idx in keep_idxs if len(g.out_edges(idx)) != 0]
        keep_edges = list(map(idx_node_map.get, keep_edges))
        for k, df in attribs_dict.items():
            attribs_dict[k] = df[df.index.isin(keep_edges)]

        binary_mtx = binary_mtx[np.ix_(keep_idxs,keep_idxs)]

        return ts_qobs_df, ts_exos_dict, attribs_dict, binary_mtx


    ############################################################################

    def compute_similarity(self, method: str, **kwargs):
        if method == 'distance':
            finite_dist = self.distance_connectivity.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.distance_connectivity, sigma)
        elif method == 'stcn':
            sigma = 10
            return gaussian_kernel(self.distance_connectivity, sigma)
        elif method == 'binary':
            return self.binary_connectivity

    def get_datetime_features(self, df):
        day = 24 * 60 * 60
        year = 365.2425 * day
        mapping = year * 10 ** 9
        ts_seconds = df.index.view(np.int64)

        datetime_feats = pd.DataFrame(index=df.index)
        datetime_feats['year_sin'] = np.sin(ts_seconds * (2 * np.pi / mapping))
        datetime_feats['year_cos'] = np.cos(ts_seconds * (2 * np.pi / mapping))

        return datetime_feats

    def get_splitter(self, method=None, val_start=None):
        if method == 'at_datetime' and val_start is not None:
            val_start = tuple(map(int, val_start.split('-')))
            return AtTimeStepSplitter(first_val_ts=val_start,
                                      first_test_ts=self.test_start)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = Ergene()
    plt.imshow(dataset.mask, aspect='auto')
    plt.show()
