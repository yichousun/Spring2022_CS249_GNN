import pickle
import numpy as np
from collections import Sequence


class EdgeTable(object):
    def __init__(self, srcs, dests, edge_data=None, n_nodes=None):
        """
        srcs: np.array of shape [n,], dtype np.int/np.int32
        dsts: np.array of shape [n,], dtype np.int/np.int32
        """
        self.src = srcs
        self.dest = dests
        self.data = edge_data
        self.n_nodes = 0 if n_nodes is None else n_nodes
        self.n_nodes = max(np.max(srcs)+1, np.max(dests)+1, self.n_nodes)
        self._src_index = None
        self._dest_index = None
        self._src_unique = None
        self._dest_unique = None
        self.number_of_nodes = len(np.unique(self.src))
    
    @property
    def src_unique(self):
        if self._src_unique is None:
            self._src_unique = np.unique(self.src)
        return self._src_unique

    @property
    def dest_unique(self):
        if self._dest_unique is None:
            self._dest_unique = np.unique(self.dest)
        return self._dest_unique

    def save(self, path):
        data = {key: getattr(self, key) for key in ['src', 'dest', 'data', 'n_nodes', '_src_index', '_dest_index']}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def save_py2(self, path):
        data = {key: getattr(self, key) for key in ['src', 'dest', 'data', 'n_nodes', '_src_index', '_dest_index']}
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=2)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        ret = EdgeTable(data['src'], data['dest'], data['data'], data['n_nodes'])
        ret._src_index = data['_src_index']
        ret._dest_index = data['_dest_index']

        return ret

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.dest[idx], (None if self.data is None else self.data[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def build_src_index(self):
        self._src_index = [[] for _ in range(self.n_nodes)]
        for i, s in enumerate(self.src):
            self._src_index[s].append(i)

    def build_dest_index(self):
        self._dest_index = [[] for _ in range(self.n_nodes)]
        for i, d in enumerate(self.dest):
            self._dest_index[d].append(i)

    def check_index_build(self, index_type='src'):
        if index_type == 'src' and self._src_index is None:
            raise ValueError('src Index is not built!')
        if index_type == 'dest' and self._dest_index is None:
            raise ValueError('dest Index is not built!')

    def get_edges_by_src(self, src, fields=None):
        try:
            index = self._src_index[src]
        except TypeError:
            print(src)

        if fields is None:
            fields = ('src', 'dest', 'data')
        sources = tuple(getattr(self, f) for f in fields)
        return tuple(s[index] if s is not None else None for s in sources)
    
    def get_edges_by_srcs(self, srcs, fields=None):
        """
        srcs: a int, or a list/1d np.array of src idx
        """
        self.check_index_build('src')
        if isinstance(srcs, Sequence) or isinstance(srcs, np.ndarray):
            return [self.get_edges_by_src(s, fields=fields) for s in srcs]
        return self.get_edges_by_src(srcs, fields=fields)

    def reverse(self):
        # print(self.n_nodes)
        tbl = EdgeTable(self.dest, self.src, edge_data=self.data, n_nodes=self.n_nodes)
        if self._dest_index is None:
            self.build_dest_index()
        tbl._src_index = self._dest_index
        tbl._dest_index = self._src_index
        return tbl

    def nodes(self, with_data=False):
        if with_data:
            for i in range(self.number_of_nodes):
                yield i, self._node_table[i]
        else:
            for i in self.src_unique:
                yield i

