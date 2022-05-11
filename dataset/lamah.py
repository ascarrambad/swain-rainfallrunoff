import os
import pickle
import tarfile
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


class LamaH(PandasDataset):
    url = "https://zenodo.org/record/5153305/files/2_LamaH-CE_daily.tar.gz?download=1"

    similarity_options = {'distance', 'stcn', 'binary'}
    temporal_aggregation_options = None
    spatial_aggregation_options = None

    def __init__(self, discard_disconnected_components=True, root='./data/', freq='1D'):
        # Set root path
        self.root = root

        # load dataset
        ts_qobs_df, ts_qobs_mask, ts_exos_dict, attribs_dict, dists_mtx, binary_mtx = self.load(discard_disconnected_components)

        super().__init__(dataframe=ts_qobs_df,
                         mask=ts_qobs_mask,
                         exogenous=ts_exos_dict,
                         attributes=attribs_dict,
                         freq=freq,
                         similarity_score="binary",
                         name="LamaH-CE")

        self.distance_connectivity = dists_mtx
        self.binary_connectivity = binary_mtx
        self.test_start = datetime(2005,1,1)

    @property
    def raw_files_paths(self):
        return ['./data/2_LamaH-CE_daily.tar.gz']

    @property
    def required_file_names(self):
        return ['LamaH-CE/lamah_qobs.csv',
                'LamaH-CE/lamah_exos.pickle',
                'LamaH-CE/lamah_attribs.pickle',
                'LamaH-CE/lamah_dists.npy',
                'LamaH-CE/lamah_hierarchy.npy']

    def download(self) -> None:
        download_url(self.url, self.root_dir)

    def build(self) -> None:
        self.maybe_download()
        logger.info('Building data...')

        # Extract downloaded file(s)
        if not files_exist(['./data/LamaH-CE']):
            for f_path in self.raw_files_paths:
                with tarfile.open(f_path) as file:
                    file.extractall(os.path.join(self.root_dir, 'LamaH-CE'))
                # os.unlink(f_path)

        # Build discharge and exogenous data
        gauges_path = os.path.join(self.root_dir, 'LamaH-CE/D_gauges/2_timeseries/daily/')
        exos_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/2_timeseries/daily/')

        ts_qobs_dfs = []
        ts_exos_dfs = []

        gauges_files = set(os.listdir(gauges_path))
        exos_files = set(os.listdir(exos_path))
        gauge_hierarchy_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Gauge_hierarchy.csv')
        gauge_hierarchy = pd.read_csv(gauge_hierarchy_path, sep=';').set_index('ID')
        isolated_nodes = gauge_hierarchy[(gauge_hierarchy['NEXTDOWNID']==0) & (gauge_hierarchy['HIERARCHY']==1)].index.tolist()
        isolated_nodes = set(map(lambda x: f'ID_{x}.csv', isolated_nodes))

        common_files = list(gauges_files.intersection(exos_files).difference(isolated_nodes))
        excluded_files = list(gauges_files.difference(exos_files))

        if len(excluded_files) > 0:
            logger.warning(f'These {len(excluded_files)} gauges do not have a matching basin, and have been excluded:\n{excluded_files}')

        if len(list(isolated_nodes)) > 0:
            logger.warning(f'These {len(list(isolated_nodes))} gauges are isolated and have been excluded:\n{isolated_nodes}')

        # Build data & exogenous time series
        for i, gauge_filename in enumerate(common_files):
            gauge_id = int(gauge_filename.split('.')[0][3:])
            print(f'Now processing {gauge_filename} ({i+1}/{len(common_files)})', end='\r')

            gauge_path = os.path.join(gauges_path, gauge_filename)
            gauge_df = pd.read_csv(gauge_path, sep=';')
            date_keys = ['YYYY','MM','DD']

            gauge_df['date'] = pd.to_datetime(gauge_df[date_keys].rename(columns=dict(zip(date_keys, ['year', 'month', 'day']))))
            gauge_df = gauge_df.drop(columns=date_keys + ['ckhs', 'qceq', 'qcol'])
            gauge_df = gauge_df.set_index('date')
            gauge_df = gauge_df.rename(columns={'qobs': gauge_id})
            
            ts_qobs_dfs.append(gauge_df)

            ######

            exo_path = os.path.join(exos_path, gauge_filename)
            exo_df = pd.read_csv(exo_path, sep=';')

            exo_df['date'] = pd.to_datetime(exo_df[date_keys].rename(columns=dict(zip(date_keys, ['year', 'month', 'day']))))
            exo_df = exo_df.drop(columns=date_keys + ['DOY'])
            exo_df = exo_df.set_index('date')
            exo_df = exo_df.filter(gauge_df.index, axis=0)
            exo_df = pd.concat([exo_df, self.get_datetime_features(exo_df)], axis=1)

            exo_df.columns = pd.MultiIndex.from_product(iterables=[[gauge_id], exo_df.columns],
                                                        names=('id', 'u'))
            ts_exos_dfs.append(exo_df)

        ts_qobs_df = pd.concat(ts_qobs_dfs, axis=1)
        ts_exos_df = pd.concat(ts_exos_dfs, axis=1).astype(np.float32, errors='ignore')

        ts_qobs_df = ts_qobs_df.sort_index(axis=1)
        ts_exos_df = ts_exos_df.sort_index(axis=1, level='id')

        ts_exos_dict = {'u': ts_exos_df}

        # Build distances matrix
        keep_ids = list(ts_qobs_df.columns)
        dists_mtx = self.build_distance_matrix(keep_ids)
        hierarchy_mtx, edge_attr_df = self.build_hierarchy_matrix(keep_ids)

        # Build static attributes data
        catch_attr_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Catchment_attributes.csv')
        catch_attr_df = pd.read_csv(catch_attr_path, sep=';').set_index('ID')
        catch_attr_df = catch_attr_df.drop(columns=['hi_prec_ti', 'lo_prec_ti', 'gc_dom', 'NEXTDOWNID'])
        catch_attr_df = catch_attr_df.filter(keep_ids, axis=0).sort_index(axis=0)

        attribs_dict = {'catchment': catch_attr_df,
                        'stream': edge_attr_df}

        # Store built data
        ts_qobs_df.to_csv(os.path.join(self.root_dir, 'LamaH-CE/lamah_qobs.csv'))
        with open(os.path.join(self.root_dir, 'LamaH-CE/lamah_exos.pickle'), 'wb') as file:
            pickle.dump(ts_exos_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.root_dir, 'LamaH-CE/lamah_attribs.pickle'), 'wb') as file:
            pickle.dump(attribs_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(os.path.join(self.root_dir, 'LamaH-CE/lamah_dists.npy'), dists_mtx)
        np.save(os.path.join(self.root_dir, 'LamaH-CE/lamah_hierarchy.npy'), hierarchy_mtx)

        # self.clean_downloads()

    def load_raw(self):
        self.maybe_build()

        # Data paths
        gauges_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_qobs.csv')
        exos_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_exos.pickle')
        attribs_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_attribs.pickle')
        dists_mtx_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_dists.npy')
        binary_mtx_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_hierarchy.npy')

        # Load time-series data
        ts_qobs_df = pd.read_csv(gauges_path)
        ts_qobs_df['date'] = ts_qobs_df['date'].astype('datetime64[ns]')
        ts_qobs_df = ts_qobs_df.set_index('date').astype(np.float32, errors='ignore')
        ts_qobs_df.columns = ts_qobs_df.columns.map(int)

        with open(exos_path, 'rb') as file:
            ts_exos_dict = pickle.load(file)

        # Load attributes
        with open(attribs_path, 'rb') as file:
            attribs_dict = pickle.load(file)

        # Load distance matrix
        dists_mtx = np.load(dists_mtx_path)
        binary_mtx = np.load(binary_mtx_path)

        return ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx, binary_mtx

    def load(self, discard_disconnected_components):
        data = self.load_raw()
        if discard_disconnected_components:
            logger.info('Disconnected components have been discarded. Only the main river network has been loaded. ')
            data = self._discard_disconnected_components(*data)

        ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx, binary_mtx = data

        # Transform eventual pd.DataFrames into ndarrays
        attribs_dict.update({k: df.to_numpy() for k, df in attribs_dict.items() if isinstance(df, pd.DataFrame)})

        # Compute validity mask and fill NaNs
        ts_qobs_df = ts_qobs_df.replace({-999: np.NaN})
        ts_qobs_mask = ~np.isnan(ts_qobs_df.values)
        ts_qobs_df = ts_qobs_df.fillna(value=0, axis=1)
        ts_exos_dict['u'].fillna(value=0, axis=1, inplace=True)
        attribs_dict['catchment'] = np.nan_to_num(attribs_dict['catchment'])

        return ts_qobs_df, ts_qobs_mask, ts_exos_dict, attribs_dict, dists_mtx, binary_mtx

    def load_plotting_data(self):
        # Load gauges (nodes) attributes
        gauge_attribs_path = os.path.join(self.root_dir, 'LamaH-CE/D_gauges/1_attributes/Gauge_attributes.csv')
        gauge_attribs = pd.read_csv(gauge_attribs_path, sep=';').set_index('ID').astype(np.float32, errors='ignore')

        # Filter out unused gauges
        gauge_attribs = gauge_attribs.loc[self.nodes]
        node_idx_map = {n:i for i, n in enumerate(self.nodes)}

        return gauge_attribs, node_idx_map

    ############################################################################

    def build_distance_matrix(self, ids):
        logger.info('Building distance matrix...')

        raw_dist_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Stream_dist.csv')
        distance_df = pd.read_csv(raw_dist_path, sep=';').set_index('ID', drop=False)
        distance_df = distance_df.filter(ids, axis=0).sort_index(axis=0)
        num_sensors = len(ids)
        distances = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf

        # Builds sensor id to index map
        sensor_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

        # Fills cells in the matrix with distances
        for row in distance_df.values:
            distances[sensor_to_ind[row[0]], sensor_to_ind[row[1]]] = row[-1]

        return distances

    def build_hierarchy_matrix(self, ids):
        logger.info('Building hierarchy matrix...')

        gauge_hierarchy_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Gauge_hierarchy.csv')
        hierarchy_df = pd.read_csv(gauge_hierarchy_path, sep=';').set_index('ID', drop=False)
        hierarchy_df = hierarchy_df.filter(ids, axis=0).sort_index(axis=0)
        hierarchy_df = hierarchy_df[hierarchy_df.NEXTDOWNID != 0]
        num_sensors = len(ids)
        hierarchy = np.zeros((num_sensors, num_sensors), dtype=np.int)

        # Builds sensor id to index map
        sensor_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

        # Fills cells in the matrix with distances
        for row in hierarchy_df.values:
            hierarchy[sensor_to_ind[row[0]], sensor_to_ind[row[-1]]] = 1

        logger.info('Building edge features...')
        edge_attr_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Stream_dist.csv')
        edge_attr_df = pd.read_csv(edge_attr_path, sep=';').set_index('ID').drop(columns=['NEXTDOWNID'])

        return hierarchy, edge_attr_df

    def _discard_disconnected_components(self, ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx, binary_mtx):
        ind_to_sensor = {i: int(sensor_id) for i, sensor_id in enumerate(ts_qobs_df.columns)}
        nodes_idxs = list(range(len(ts_qobs_df.columns)))
        edge_index, _ = adj_to_edge_index(binary_mtx)

        g = nx.Graph()
        g.add_nodes_from(nodes_idxs)
        g.add_edges_from(edge_index.T)

        comps = list(nx.connected_components(g))
        main_comp_idx = np.argmax([len(c) for c in comps])
        del comps[main_comp_idx]
        secondary_nodes_idxs = list(set.union(*comps))
        secondary_nodes = [ind_to_sensor[idx] for idx in secondary_nodes_idxs]

        ts_qobs_df.drop(columns=secondary_nodes, inplace=True)
        ts_exos_dict['u'].drop(columns=secondary_nodes, level='id', inplace=True)

        attribs_dict = {k: df.drop(index=secondary_nodes, errors='ignore') for k, df in attribs_dict.items()}
        dists_mtx = np.delete(np.delete(dists_mtx, secondary_nodes_idxs, axis=0), secondary_nodes_idxs, axis=1)
        binary_mtx = np.delete(np.delete(binary_mtx, secondary_nodes_idxs, axis=0), secondary_nodes_idxs, axis=1)

        return ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx, binary_mtx


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

    dataset = LamaH()
    plt.imshow(dataset.mask, aspect='auto')
    plt.show()
