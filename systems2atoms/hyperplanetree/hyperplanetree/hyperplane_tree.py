import copy
import json
import torch
import inspect

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from ._classes import recursive_to_device

from .lineartree import LinearTreeRegressor
from .linear_combinations import LinearCombinations


class HyperplaneMixin():
    """Automatically take hyperplanes of features

    A Mixin for sklearn-like models to automatically take hyperplanes of features
    before doing anything with them. The implemented functions should cover most
    sklearn-like models.

    Parameters
    ----------
    max_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    hyperplane_weights : torch.Tensor of float or None, default = None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=2
        Maximum number of terms to use if auto-generating hyperplane weights.

    do_symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    do_scaling: bool (default: True)
        Automatically scale weights in LCs to correspond to maximum and minimum values in data

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : torch.device or str or None; default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    disable_tqdm : bool, default = False
        Disable progress bars powered by TQDM
    
    categorical_features : torch.Tensor of ints or None, default = None
        Features that should be treated as categorical and excluded from hyperplanes
    """
    
    def __init__(
            self,
            hyperplane_weights = None,
            num_terms = 2,
            do_symmetrize = True,
            do_scaling = True,
            tol_decimals = 4,
            torch_device = 'cpu',
            max_weight = 1,
            disable_tqdm = False,
            categorical_features = None,
            ):
                
        self.hyperplane_weights = hyperplane_weights
        self.num_terms = num_terms
        self.do_symmetrize = do_symmetrize
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.max_weight = max_weight
        self.do_scaling = do_scaling
        self.categorical_features = categorical_features

        self.disable_tqdm = disable_tqdm

        self.linear_combinations_transform = LinearCombinations(
            LCs = hyperplane_weights,
            num_terms = num_terms,
            do_symmetrize = do_symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_weight = max_weight,
            do_scaling = do_scaling,
            categorical_features = categorical_features
        )

    def do_lcs(self, x):
        return self.linear_combinations_transform.transform(x)
    
    def fit(self, X, y, **kwargs):
        self.original_features_count = X.shape[1]
        assert X.shape[1] >= self.num_terms, f'Cannot use {self.num_terms} terms in hyperplanes when data only has {X.shape[1]} columns!'
        if not hasattr(self, 'linear_features') or self.linear_features is None:
            self.linear_features = torch.arange(start = 0, end=len(X[0]), device = self.torch_device, dtype = torch.int)

        X = self.do_lcs(X)
    
        return super().fit(X, y, **kwargs)
    
    def predict(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().predict(X, **kwargs)
    
    def uncertainty(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().uncertainty(X, **kwargs)
    
    def apply(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().apply(X, **kwargs)
    
    def decicion_path(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().decision_path(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().predict_proba(X, **kwargs)
    
    def predict_log_proba(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().predict_log_proba(X, **kwargs)
    
    def decision_function(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().decision_function(X, **kwargs)
    
    def score(self, X, **kwargs):
        X = self.do_lcs(X)
        return super().score(X, **kwargs)
    
    def write_to_json(self, filename):
        out = {}
        out['nodes'] = copy.deepcopy(super().summary())

        for key, node in out['nodes'].items():
            if 'col' in node.keys():
                #Splitting node
                node['col'] = node['col'].item()
                node['th'] = node['th'].item()
                del node['models']

            else:
                #leaf node
                node['model'] = node['models'].__dict__
                node['model']['params'] = node['model']['params'].to('cpu').tolist()
                del node['models']

            node['loss'] = node['loss'].item()
            node['samples'] = node['samples'].item()

    
        out['hyperplanes_final_matrix'] = self.linear_combinations_transform.final_matrix.to('cpu').tolist()
        out['type'] = str(super())[36:-9]
        out['categorical_features'] = self._categorical_features.to('cpu').tolist()
        out['linear_features'] = self._linear_features.to('cpu').tolist()
        out['split_features'] = self._split_features.to('cpu').tolist()
        out['criterion'] = self.criterion
        out['max_depth'] = self.max_depth
        out['min_samples_leaf'] = self.min_samples_leaf
        out['n_features_in'] = self.n_features_in_
        out['n_targets'] = self.n_targets_

        with open(filename, 'w') as outfile:
            json.dump(out, outfile)

    def to(self, device):
        self.torch_device = device
        return recursive_to_device(self, device)


class HyperplaneTreeRegressor(HyperplaneMixin, LinearTreeRegressor):
    """A Hyperplane Tree Regressor.

    A Hyperplane Tree Regressor is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are splitted according
    simple decision rules. The goodness of slits is evaluated in gain terms
    fitting linear models in each node. This implies that the models in the
    leaves are linear instead of constant approximations like in classical
    Decision Tree.

    Parameters
    ----------
    min_samples_leaf : int or float, default=0.01
        The recommended parameter to control the size of the tree.
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least `min_samples_leaf` training samples in each of the left and
        right branches.
        The minimum valid number of samples in each leaf is 3.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_weight : int, default = 3
        The recommended parameter to control the training time of the tree
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info
        Increases training time roughly factorially

    hyperplane_weights : torch.Tensor of float or None, default = None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=2
        Maximum number of terms to use if auto-generating hyperplane weights.
        Increases training time roughly factorially

    do_symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    do_scaling: bool (default: True)
        Automatically scale weights in LCs to correspond to maximum and minimum values in data

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    categorical_features : torch.Tensor of ints or None, default = None
        Features that should be treated as categorical and excluded from hyperplanes

    criterion : str or Callable, default = 'mae'
        Loss function to use when determining splits.
        If str, must be one of ['mae', 'rmse', 'msle', and 'max_abs']

    max_depth : int, default=torch.inf
        The maximum depth of the tree considering only the splitting nodes.
        A higher value implies a higher training time.

    min_samples_split : int or float, default=6
        The minimum number of samples required to split an internal node.
        The minimum valid number of samples in each node is 6.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    max_bins : int, default=10
        The maximum number of bins to use to search the optimal split in each
        feature. Features with a small number of unique values may use less than
        ``max_bins`` bins. Increases training time roughly linearly

    min_impurity_decrease : float, default= -torch.inf
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value. Can be used to regularize the tree

    split_features : int or array-like of int, default=None
        Defines which features can be used to split on.
        All split feature indices must be in `[0, n_features)`.
        - None : All features will be used for splitting.
        - integer array-like : integer indices indicating splitting features.
        - integer : integer index indicating a single splitting feature.

    linear_features : int or array-like of int, default=None
        Defines which features are used for the linear model in the leaves.
        All linear feature indices must be in `[0, n_features)`.
        - None : All features except those in `categorical_features`
          will be used in the leaf models.
        - integer array-like : integer indices indicating features to
          be used in the leaf models.
        - integer : integer index indicating a single feature to be used
          in the leaf models.

    disable_tqdm : bool, default=False
        Disable the TQDM-powered training progress bar

    save_linear_propogation_uncertainty_parameters : bool, default=False
        Save parameters needed for uncertainty quantification by propogation
        through linear regression normal equation

    save_quadratic_uncertainty_parameters : bool, default=False
        Save parameters needed for uncertainty quantification by Lagrange
        estimation (difference from linear regression with quadratic features).

    max_batch_size : int, default = torch.int
        Maximum amount of data used to fit a split.
        Allows fitting of large datasets with limited memory

    depth_first : bool, default = True
        Make splits depth-first through the tree.
        If False, make splits breadth-first.
        Depth-first is generally recommended for the following reasons:
        1. Leaves close in index are usually close in domain
        2. More accurate training time estimation

    ridge : float, default = 1e-5
        Regularization parameter for the linear models in the leaves.
        A higher value implies a higher regularization.
    """

    def __init__(
        self,
        min_samples_leaf = 0.01,
        max_weight: int = 3,
        hyperplane_weights = None,
        num_terms: int = 2,
        do_symmetrize: bool = True,
        do_scaling: bool = True,
        tol_decimals: int = 4,
        torch_device = None,
        disable_tqdm: bool = False,
        categorical_features = None,
        criterion = 'mae',
        max_depth = torch.inf,
        min_samples_split = 6,
        max_bins: int = 10,
        min_impurity_decrease = -torch.inf,
        split_features = None,
        linear_features = None,
        save_linear_propogation_uncertainty_parameters: bool = False,
        save_quadratic_uncertainty_parameters: bool = False,
        max_batch_size = torch.inf,
        depth_first = True,
        ridge = 1e-5,

        ):
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights,
            num_terms,
            do_symmetrize,
            do_scaling,
            tol_decimals,
            torch_device,
            max_weight,
            disable_tqdm,
            categorical_features,
        )

        LinearTreeRegressor.__init__(
            self,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_bins,
            min_impurity_decrease,
            categorical_features,
            split_features,
            linear_features,
            disable_tqdm,
            save_linear_propogation_uncertainty_parameters,
            save_quadratic_uncertainty_parameters,
            max_batch_size,
            depth_first,
            ridge,
        )