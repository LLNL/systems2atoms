import copy
import torch

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from tqdm.auto import tqdm

from ._classes import TorchLinearRegression, _LinearTree, _LinearForest, _predict_branch

class LinearTreeRegressor(_LinearTree, RegressorMixin):
    """A Linear Tree Regressor.

    A Linear Tree Regressor is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are splitted according
    simple decision rules. The goodness of slits is evaluated in gain terms
    fitting linear models in each node. This implies that the models in the
    leaves are linear instead of constant approximations like in classical
    Decision Tree.

    Parameters
    ----------
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

    min_samples_leaf : int or float, default=0.01
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

    max_bins : int, default=25
        The maximum number of bins to use to search the optimal split in each
        feature. Features with a small number of unique values may use less than
        ``max_bins`` bins. Increases training time roughly linearly

    min_impurity_decrease : float, default= -torch.inf
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value. Can be used to regularize the tree

    categorical_features : int or torch.Tensor of int, default=None
        Indicates the categorical features.
        All categorical indices must be in `[0, n_features)`.
        Categorical features are used for splits but are not used in
        model fitting.
        More categorical features imply a higher training time.
        - None : no feature will be considered categorical.
        - integer tensor : integer indices indicating categorical
          features.
        - integer : integer index indicating a categorical
          feature.

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
        Regularization strength for the linear models in the leaves.
        A higher value implies a higher regularization strength.
    """

    def __init__(
        self,
        criterion='mae',
        max_depth=torch.inf,
        min_samples_split=6,
        min_samples_leaf=0.01,
        max_bins=25,
        min_impurity_decrease=-torch.inf,
        categorical_features=None,
        split_features=None,
        linear_features=None,
        disable_tqdm = False,
        save_linear_propogation_uncertainty_parameters = False,
        save_quadratic_uncertainty_parameters = False,
        max_batch_size = torch.inf,
        depth_first = True,
        ridge = 1e-5,
        ):

        self.base_estimator = TorchLinearRegression()

        super().__init__(
            base_estimator = self.base_estimator,
            criterion = criterion,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            max_bins = max_bins,
            min_impurity_decrease = min_impurity_decrease,
            categorical_features = categorical_features,
            split_features = split_features,
            linear_features = linear_features,
            disable_tqdm = disable_tqdm,
            save_linear_propogation_uncertainty_parameters = save_linear_propogation_uncertainty_parameters,
            save_quadratic_uncertainty_parameters = save_quadratic_uncertainty_parameters,
            max_batch_size = max_batch_size,
            depth_first = depth_first,
            ridge = ridge,
            )

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Tree of a linear estimator from the training
        set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        # Convert data (X is required to be 2d and indexable)

        if not hasattr(self, 'original_features_count'):
            self.original_features_count = X.shape[1]

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        y_shape = y.shape
        self.n_targets_ = y_shape[1] if len(y_shape) > 1 else 1
        if self.n_targets_ < 2:
            y = y.ravel()
        self._fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if
            multitarget regression.
            The predicted values.
        """
        check_is_fitted(self, attributes='_nodes')

        if self.n_targets_ > 1:
            pred = torch.zeros((X.shape[0], self.n_targets_), device = X.device, dtype = X.dtype)
        else:
            pred = torch.zeros(X.shape[0], device = X.device, dtype = X.dtype)

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            pred[mask] = L.model.predict(X[mask][:, self._linear_features]).reshape(pred[mask].shape)

        return pred
    
    def uncertainty(self, X, **kwargs):
        """Obtain uncertainty on X

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : tensor of shape (n_samples, ) or also (n_samples, n_targets) if
            multitarget regression.
            The predicted uncertainties
        """
        check_is_fitted(self, attributes='_nodes')

        if self.n_targets_ > 1:
            unc = torch.zeros((X.shape[0], self.n_targets_))
        else:
            unc = torch.zeros(X.shape[0], device = X.device)

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            unc[mask] = L.model.uncertainty(X[mask][:, self._linear_features], **kwargs)

        return unc


class LinearForestRegressor(BaseEstimator, RegressorMixin):
    parameter_docstring = ''
    def __init__(
        self,
        base_linear_tree,
        number_of_trees = 100,
        features_per_tree = 1,
        data_per_tree = 100,
        random_seed = 42,
        disable_tqdm = False,
    ):
        self.random_seed = random_seed
        self.data_per_tree = data_per_tree
        self.features_per_tree = features_per_tree
        self.disable_tqdm = disable_tqdm

        self.trees = [copy.deepcopy(base_linear_tree) for i in range(number_of_trees)]

    def fit(self, X, y, sample_weights = None):
        torch.manual_seed(self.random_seed)
        for tree in tqdm(self.trees, disable = self.disable_tqdm):
            if self.features_per_tree is not None:
                if tree.split_features is None:
                    tree.split_features = torch.randperm(len(X[0]), device = X.device)[:self.features_per_tree]
                else:
                    tree.split_features = tree.split_features[torch.randperm(len(tree.split_features), device = X.device)[:self.features_per_tree]]

            if self.data_per_tree is not None:
                random_rows = torch.randperm(len(X), device = X.device)[:self.data_per_tree]
            else:
                random_rows = torch.range(0, len(X)-1, device = X.device).type(torch.int)

            if sample_weights is not None:
                tree.fit(X[random_rows], y[random_rows], sample_weights[random_rows])

            else:
                tree.fit(X[random_rows], y[random_rows])

    def predict(self, X):
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        return torch.mean(predictions, dim = 0)

    def uncertainty(self, X):
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        return torch.std(predictions, dim = 0)