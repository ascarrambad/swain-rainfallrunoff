import os
import pickle
import tarfile
from functools import reduce
from collections import defaultdict

import numpy as np
import pandas as pd

from tsl import logger
from tsl.utils import download_url
from tsl.utils.python_utils import files_exist
from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes import PandasDataset


class LamaH(PandasDataset):
    url = "https://zenodo.org/record/5153305/files/2_LamaH-CE_daily.tar.gz?download=1"

    similarity_options = {'distance', 'stcn'}
    temporal_aggregation_options = None
    spatial_aggregation_options = None

    def __init__(self, root='./data/', freq='1D'):
        # Set root path
        self.root = root

        # load dataset
        ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx, ts_qobs_mask = self.load()
        attribs_dict['dist'] = dists_mtx

        super().__init__(dataframe=ts_qobs_df,
                         mask=ts_qobs_mask,
                         exogenous=ts_exos_dict,
                         attributes=attribs_dict,
                         freq=freq,
                         similarity_score="distance",
                         name="LamaH-CE")

    @property
    def raw_files_paths(self):
        return ['./data/2_LamaH-CE_daily.tar.gz']

    @property
    def required_file_names(self):
        return ['LamaH-CE/lamah_qobs.csv',
                'LamaH-CE/lamah_exos.pickle',
                'LamaH-CE/lamah_attribs.pickle',
                'LamaH-CE/lamah_dists.npy']

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
        common_files = list(gauges_files.intersection(exos_files))
        excluded_files = list(gauges_files.difference(exos_files))

        if len(excluded_files) > 0:
            logger.warning(f'These {len(excluded_files)} gauges do not have a matching basin, and have been excluded:\n{excluded_files}')


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

            # reindex = pd.MultiIndex.from_product(iterables=[[gauge_id], exo_df.columns],
            #                                      names=('id', 'date'))
            # ts_exos_df = ts_exos_df.append(exo_df.set_index(reindex))
            exo_df.columns = pd.MultiIndex.from_product(iterables=[[gauge_id], exo_df.columns],
                                                        names=('id', 'exogenous'))
            ts_exos_dfs.append(exo_df)

        ts_qobs_df = pd.concat(ts_qobs_dfs, axis=1)
        ts_exos_df = pd.concat(ts_exos_dfs, axis=1).astype(np.float32, errors='ignore')

        ts_qobs_df = ts_qobs_df.sort_index(axis=1)
        ts_exos_df = ts_exos_df.sort_index(axis=1, level='id')

        # ts_exos_dict = {id: df.unstack('id') for id, df in ts_exos_df.to_dict('series').items()}
        ts_exos_dict = {'u': ts_exos_df}

        # Build attributes data
        # attrib_gauges_path = os.path.join(self.root_dir, 'LamaH-CE/D_gauges/1_attributes/Gauge_attributes.csv')
        attrib_exos_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Catchment_attributes.csv')

        # attrib_gauges_df = pd.read_csv(attrib_gauges_path, sep=';').set_index('ID')
        attrib_exos_df = pd.read_csv(attrib_exos_path, sep=';').iloc[:, :-1].set_index('ID')

        attribs_dict = {'u': attrib_exos_df}

        # Build distances matrix
        ids = list(ts_qobs_df.columns)
        dists_mtx = self.build_distance_matrix(ids)

        # Store built data
        ts_qobs_df.to_csv(os.path.join(self.root_dir, 'LamaH-CE/lamah_qobs.csv'))
        with open(os.path.join(self.root_dir, 'LamaH-CE/lamah_exos.pickle'), 'wb') as file:
            pickle.dump(ts_exos_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.root_dir, 'LamaH-CE/lamah_attribs.pickle'), 'wb') as file:
            pickle.dump(attribs_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(os.path.join(self.root_dir, 'LamaH-CE/lamah_dists.npy'), dists_mtx)

        # self.clean_downloads()

    def load_raw(self):
        self.maybe_build()

        # Data paths
        gauges_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_qobs.csv')
        exos_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_exos.pickle')
        attribs_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_attribs.pickle')
        dists_mtx_path = os.path.join(self.root_dir, 'LamaH-CE/lamah_dists.npy')

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

        return ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx

    def load(self):
        ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx = self.load_raw()

        # Compute validity mask and fill NaNs
        ts_qobs_mask = ~np.isnan(ts_qobs_df.values)
        ts_qobs_df = ts_qobs_df.fillna(value=0, axis=1)
        ts_exos_dict['u'].fillna(value=0, axis=1, inplace=True)

        return ts_qobs_df, ts_exos_dict, attribs_dict, dists_mtx, ts_qobs_mask

    def build_distance_matrix(self, ids):
        logger.info('Building distance matrix...')
        raw_dist_path = os.path.join(self.root_dir, 'LamaH-CE/B_basins_intermediate_all/1_attributes/Stream_dist.csv')
        distances = pd.read_csv(raw_dist_path, sep=';')
        num_sensors = len(ids)
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf

        # Builds sensor id to index map
        sensor_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

        # Fills cells in the matrix with distances
        for row in distances.values:
            dist[sensor_to_ind[row[0]], sensor_to_ind[row[1]]] = row[-1]

        return dist

    def compute_similarity(self, method: str, **kwargs):
        if method == 'distance':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
        elif method == 'stcn':
            sigma = 10
            return gaussian_kernel(self.dist, sigma)

    # Old methods, still to be checked

    # def get_datetime_dummies(self):
    #     df = self.dataframe()
    #     df['day'] = df.index.weekday
    #     df['hour'] = df.index.hour
    #     df['minute'] = df.index.minute
    #     dummies = pd.get_dummies(df[['day', 'hour', 'minute']],
    #                              columns=['day', 'hour', 'minute'])
    #     return dummies.values

    def get_datetime_features(self, df):
        day = 24 * 60 * 60
        year = 365.2425 * day
        mapping = year * 10 ** 9
        ts_seconds = df.index.view(np.int64)

        datetime_feats = pd.DataFrame(index=df.index)
        datetime_feats['year_sin'] = np.sin(ts_seconds * (2 * np.pi / mapping))
        datetime_feats['year_cos'] = np.cos(ts_seconds * (2 * np.pi / mapping))

        return datetime_feats

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window],
                idx[test_start:]]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = LamaH()
    plt.imshow(dataset.mask, aspect='auto')
    plt.show()
