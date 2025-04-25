import copy
from typing import Callable
import warnings
import json
import numbers
import scipy.sparse as sp
import torch
import sklearn
from copy import deepcopy

from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

from tqdm.auto import tqdm


# Loss Criteria
def mae(y, yh, weights = None, dim = -1, **largs):
    if weights is None:
        return torch.mean(torch.abs(y-yh), dim = dim)
    else:
        return torch.mean(weights*torch.abs(y-yh), dim = dim)

def rmse(y, yh, weights = None, dim = -1, **largs):
    if weights is None:
        return torch.sqrt(torch.mean((y-yh)**2,dim=dim))
    else:
        return torch.sqrt(torch.mean(weights * (y-yh)**2, dim = dim))

def msle(y, yh, weights = None, dim = -1, **largs):
    if weights is not None:
        return torch.mean(weights * (torch.square(torch.log10(y.clip(1e-6) + 1) - torch.log10(yh.clip(1e-6) + 1))), dim = -1)
    else:
        return torch.mean(torch.square(torch.log10(y.clip(1e-6) + 1) - torch.log10(yh.clip(1e-6) + 1)), dim = -1)

def max_abs(y, yh, weights = None, dim = -1, **largs):
    return torch.max(torch.abs(y - yh), dim = dim)[0]

criteria = {
    'mae': mae,
    'rmse': rmse,
    'msle': msle,
    'max_abs': max_abs,
}


def compute_theta(X, y, ridge = 1e-5):
    """Compute linear regression parameters using torch.linalg.lstsq """
    c, b, s, f = X.shape  # columns, bins, samples, features
    t = y.shape[-1]  # number of targets

    X_reshaped = X.permute(0, 1, 3, 2).reshape(c * b, f, s)
    XtX = torch.bmm(X_reshaped, X_reshaped.transpose(1, 2)).view(c, b, f, f)

    y_reshaped = y.reshape(c * b, s, t)
    Xty = torch.bmm(X_reshaped, y_reshaped).reshape(c, b, f, t)

    # Add ridge regularization to handle small data
    XtX += ridge * torch.eye(f, device=X.device).unsqueeze(0).unsqueeze(0)

    theta = torch.linalg.lstsq(XtX, Xty)[0]

    return theta


def _map_node(X, feat, direction, split):
    """Utility to map samples to nodes"""
    if direction == 'L':
        mask = (X[:, feat] <= split)
    else:
        mask = (X[:, feat] > split)

    return mask


def _predict_branch(X, branch_history, mask=None):
    """Utility to map samples to branches"""

    if mask is None:
        mask = torch.tensor([True], device = X.device).repeat(X.shape[0])

    for node in branch_history:
        mask = torch.logical_and(_map_node(X, *node), mask)

    return mask

def recursive_to_device(_self, device):
    """Utility to move a model to another device"""
    if isinstance(_self, torch.Tensor):
        _self = _self.to(device)

    elif isinstance(_self, tuple):
        _self = tuple((recursive_to_device(x, device) for x in _self))

    elif isinstance(_self, list):
        _self = [recursive_to_device(x, device) for x in _self]

    elif isinstance(_self, dict):
        for key, value in _self.items():
            recursive_to_device(value, device)

    elif hasattr(_self, '__dict__'):
        for key, attr in _self.__dict__.items():
            if hasattr(attr, 'to'):
                setattr(_self, key, attr.to(device))

            elif isinstance(attr, dict):
                for key2, value in attr.items():
                    recursive_to_device(value, device)

            elif isinstance(attr, list):
                setattr(_self, key, [recursive_to_device(x, device) for x in attr])

            elif isinstance(attr, tuple):
                setattr(_self, key, tuple((recursive_to_device(x, device) for x in attr)))

    return _self

class Node:
    def __init__(self, id=None, threshold=[],
                 parent=None, children=None,
                 n_samples=None, w_loss=None,
                 loss=None, model=None, classes=None):
        self.id = id
        self.threshold = threshold
        self.parent = parent
        self.children = children
        self.n_samples = n_samples
        self.w_loss = w_loss
        self.loss = loss
        self.model = model
        self.classes = classes

class TorchLinearRegression(LinearRegression):
    """Linear regressor using PyTorch linear algebra operations
        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
            The training input samples.

        y : torch.Tensor of shape (n_samples, )
            The target values

        sample_weight : torch.Tensor of shape (n_samples, ) or None, default=None
            Sample weights. If None, then samples are equally weighted.

        save_linear_propogation_uncertainty_parameters : bool, default=False
            Save parameters needed for uncertainty quantification by propogation
            through linear regression normal equation

        save_quadratic_uncertainty_parameters : bool, default=False
            Save parameters needed for uncertainty quantification by Lagrange
            estimation (difference from linear regression with quadratic features).

        Returns
        -------
        None
        """

    def __init__(self):
        super().__init__()
        self.scale_weight = 1
        self.scale_offset = 0
        self.target_scale_weight = 1
        self.target_scale_offset = 0

    def scale(self, x, y = None):
        if y is not None:
            return x * self.scale_weight + self.scale_offset, y * self.target_scale_weight + self.target_scale_offset
        else:
            return x * self.scale_weight + self.scale_offset

    def fit(
        self,
        x,
        y,
        sample_weight = None,
        save_linear_propogation_uncertainty_parameters = False,
        save_quadratic_uncertainty_parameters = False,
        rescale = False,
        ridge = 1e-5,
        ):

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        if rescale:
            epsilon = 1e-8  # Small value to avoid division by zero
            range_x = torch.max(x, dim=0)[0] - torch.min(x, dim=0)[0]
            range_x[range_x < epsilon] = epsilon
            self.scale_weight = 1 / range_x
            self.scale_offset = -torch.min(x, dim=0)[0] * self.scale_weight
            
            range_y = torch.max(y) - torch.min(y)
            range_y[range_y < epsilon] = epsilon
            self.target_scale_weight = 1 / range_y
            self.target_scale_offset = -torch.min(y) * self.target_scale_weight

        x, y = self.scale(x, y)
        
        if self.fit_intercept:
            x = torch.hstack((torch.ones((len(x),1), device = x.device), x))

        if sample_weight is None:
            XTX = x.T @ x

            # Add ridge regularization
            XTX += ridge * torch.eye(len(XTX), device = x.device)

            self.params = torch.linalg.lstsq(XTX, x.T @ y)[0]
        else:
            weighted_x = sample_weight @ x
            XTX = x.T @ weighted_x

            # Add ridge regularization
            XTX += ridge * torch.eye(len(XTX), device = x.device)

            self.params = torch.linalg.lstsq(XTX, x.T @ sample_weight @ y)[0]

        if save_linear_propogation_uncertainty_parameters:
            self.n = len(y)
            self.x_mean = torch.mean(x[:, 1:], axis=0)
            self.x_var = torch.var(x[:, 1:], axis=0)
            
            residuals = x @ self.params - y
            self.mse = torch.mean(residuals**2)

        if save_quadratic_uncertainty_parameters:
            x_expanded = torch.einsum('ij,ik->ijk', x, x)   
            mask = torch.triu(torch.ones_like(x_expanded, dtype = torch.bool))
            x_triu = x_expanded[mask]
            x_expanded = x_triu.reshape((len(x), -1))[:, 1:]

            self.quad_params = torch.linalg.lstsq(x_expanded.T @ x_expanded, x_expanded.T @ y)[0]

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        x = self.scale(x)

        if self.fit_intercept:
            x = torch.hstack((torch.ones((len(x),1), device = x.device), x))
        return x @ self.params
    
    def to(self, device):
        return recursive_to_device(self, device)
    
    
    def linprop_uncertainty(self, x):
        x = self.scale(x)

        return torch.sqrt(
            self.mse * (1/self.n) + torch.sum((x - self.x_mean)**2 / ((self.n - 1) * self.x_var), axis=1)
        )
    
    def quad_uncertainty(self, x):
        x = self.scale(x)

        x = torch.hstack((torch.ones((len(x),1), device = x.device), x))
        lin_pred = x @ self.params

        x_expanded = torch.einsum('ij,ik->ijk', x, x)   
        mask = torch.triu(torch.ones_like(x_expanded, dtype = torch.bool))
        x_triu = x_expanded[mask]
        x_expanded = x_triu.reshape((len(x), -1))[:, 1:]

        quad_pred = x_expanded @ self.quad_params

        return torch.abs(lin_pred - quad_pred)
    
    def uncertainty(self, x, method = 'linprop'):
        if method  == 'linprop':
            return self.linprop_uncertainty(x)
        elif method == 'quadratic':
            return self.quad_uncertainty(x)
        elif method == 'sum':
            return self.linprop_uncertainty(x) + self.quad_uncertainty(x)
        else:
            raise NotImplementedError(f'Uncertainty method {method} is not known.')
    
    # intercept_ and coef_ with numpy types for sklearn compatibility
    @property
    def intercept_(self):
        if isinstance(self.scale_offset, torch.Tensor):
            return (self.params[0] * self.target_scale_weight + self.target_scale_offset - torch.sum(self.params[1:] * self.scale_offset * self.target_scale_weight)).numpy()
        else:
            return (self.params[0] * self.target_scale_weight + self.target_scale_offset - torch.sum(self.params[1:] * torch.tensor(self.scale_offset) * self.target_scale_weight)).numpy()
    
    @property
    def coef_(self):
        if isinstance(self.scale_weight, torch.Tensor):
            return (self.params[1:] * self.scale_weight * self.target_scale_weight).numpy().T
        else:
            return (self.params[1:] * torch.tensor(self.scale_weight) * self.target_scale_weight).numpy().T


class _LinearTree(BaseDecisionTree):
    """Base class for Linear Tree meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator = TorchLinearRegression(), *, criterion, max_depth,
                 min_samples_split, min_samples_leaf, max_bins,
                 min_impurity_decrease, categorical_features,
                 split_features, linear_features, disable_tqdm,
                 save_linear_propogation_uncertainty_parameters,
                 save_quadratic_uncertainty_parameters,
                 max_batch_size, depth_first, ridge):

        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.min_impurity_decrease = min_impurity_decrease
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.disable_tqdm = disable_tqdm
        self.save_linear_propogation_uncertainty_parameters = save_linear_propogation_uncertainty_parameters
        self.save_quadratic_uncertainty_parameters = save_quadratic_uncertainty_parameters
        self.max_batch_size = max_batch_size
        self.depth_first = depth_first
        self.ridge = ridge
        
        if isinstance(criterion, Callable):
            self.loss_func = criterion
        else:
            self.loss_func = criteria.get(criterion)
            if self.loss_func is None:
                raise NotImplementedError(f'Unknown loss function "{criterion}". Consider passing a callable function as criterion.')

    def _split(self, X, y,
               weights=None,
               loss=None, min_samples_leaf=3):
        """Evaluate optimal splits in a given node (in a specific partition of
        X and y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            The target values (class labels in classification, real numbers in
            regression).

        weights : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        loss : float, default=None
            The loss of the parent node. A split is computed if the weighted
            loss sum of the two children is lower than the loss of the parent.
            A None value implies the first fit on all the data to evaluate
            the benefits of possible future splits.

        min_samples_leaf : int, defualt = 3
            Minimum number of samples in each leaf for the split to succeed. 
            Can be different from the tree-wide value if this function is
            applied to a subset (batch) of the data.

        Returns
        -------
        split_t : float
            Optimal threshold to split the data

        split_col : int
            Optimal column on which the threshold split should be a applied

        left_node : Node
            Contains the model, loss, and number of samples for the left split

        right_node : Node
            Contains the model, loss, and number of samples for the right split
        """
        # Ensure there is enough data to split
        if len(X) < min_samples_leaf * 2:
            return None, None, None, None

        # Ensure X has a bias term (column of ones)
        X = torch.cat([torch.ones(X.shape[0], 1, device = X.device), X], dim=-1)
        linear_features = torch.tensor([0, *(self._linear_features+1)], device = X.device)

        # Determine quantiles for all features based on the number of bins
        X_nb = X[:, 1:][:, self._split_features]  # Exclude the bias term
        quantiles = torch.linspace(0, 1, self.max_bins + 1, device = X.device, dtype = X.dtype)
        thresholds = torch.quantile(X_nb, quantiles, dim=0)[1:-1]

        #Split the data into two subsets for each threshold using broadcasting
        X_features = X_nb.unsqueeze(0)  # Shape: (1, N, F)

        # Create masks for below and above thresholds
        mask_below = (X_features <= thresholds.unsqueeze(1)).float()
        mask_above = 1-mask_below

        # Identify thresholds where min_samples_leaf is not satisfied in both leaves
        valid_thresholds = (mask_below.sum(dim=1) >min_samples_leaf) & (mask_above.sum(dim=1) > min_samples_leaf)

        if not torch.any(valid_thresholds):
            return None, None, None, None
        
        # Mask X and y tensors
        X_below = torch.einsum('sf,bsc->cbsf', X, mask_below)
        X_above = torch.einsum('sf,bsc->cbsf', X, mask_above)
        y_below = torch.einsum('st,bsc->cbst', y, mask_below)
        y_above = torch.einsum('st,bsc->cbst', y, mask_above)

        # Compute theta (linear regression parameters) for below and above thresholds
        theta_below = compute_theta(X_below[:, :, :, linear_features], y_below, ridge=self.ridge)
        theta_above = compute_theta(X_above[:, :, :, linear_features], y_above, ridge=self.ridge)

        # Make predictions
        y_pred_below = torch.einsum('cbsf, cbft -> cbst', X_below[:, :, :, linear_features], theta_below)
        y_pred_above = torch.einsum('cbsf, cbft -> cbst', X_above[:, :, :, linear_features], theta_above)

        y_pred = torch.einsum('cbst, bsc -> cbst', y_pred_below, mask_below) + \
             torch.einsum('cbst, bsc -> cbst', y_pred_above, mask_above)

        # Calculate error
        overall_error = self.loss_func(y, y_pred, dim = 2).permute(2, 1, 0)

        err_below = self.loss_func(y_below, y_pred_below, dim = 2).permute(2, 1, 0)
        err_above = self.loss_func(y_above, y_pred_above, dim = 2).permute(2, 1, 0)

        n_below = mask_below.sum(dim=1)
        n_above = mask_above.sum(dim=1)

        # Pool the error across the targets
        overall_error = overall_error.sum(dim=0)
        err_below = err_below.sum(dim=0)
        err_above = err_above.sum(dim=0)

        overall_error = torch.nan_to_num(overall_error, 2*torch.max(overall_error))
        overall_error[~valid_thresholds] += torch.inf

        # Identify the threshold with the lowest overall error
        best_threshold_idx, best_feature_idx = torch.unravel_index(torch.argmin(overall_error),overall_error.shape)
        lowest_error = overall_error[best_threshold_idx, best_feature_idx].item()

        if loss - lowest_error < self.min_impurity_decrease:
            # No valid split decreases error by self.min_impurity_decrease
            return None, None, None, None
        else:
            # Valid split found
            split_col = best_feature_idx.item()
            split_t = thresholds[best_threshold_idx, best_feature_idx].item()

            # Create TorchLinearRegressions for above and below
            below_mask = X[:, split_col+1] <= split_t
            above_mask = ~below_mask

            model_left = copy.deepcopy(self.base_estimator)
            model_right = copy.deepcopy(self.base_estimator)
            
            # Copy regression weights from computed "theta"
            # Will be re-fitted at the end of _LinearTree.fit() if needed
            model_left.params = theta_below[best_feature_idx, best_threshold_idx]
            model_right.params = theta_above[best_feature_idx, best_threshold_idx]
            
            # Compute weighted loss
            n_left = len(X[below_mask])
            n_right = len(X[above_mask])

            loss_left = err_below[best_threshold_idx, best_feature_idx].item()
            loss_right = err_above[best_threshold_idx, best_feature_idx].item()

            if weights is not None:
                wloss_left = loss_left * (weights[~below_mask].sum() / weights.sum())
                wloss_right = loss_right * (weights[~above_mask].sum() / weights.sum())
            else:
                wloss_left = loss_left * n_left / len(X)
                wloss_right = loss_right * n_right / len(X)

            left_node = (model_left, loss_left, wloss_left, n_left, {'classes': None})
            right_node = (model_right, loss_right, wloss_right, n_right, {'classes': None})

        return split_t, split_col, left_node, right_node
    
    def _grow(self, X, y, weights=None):
        """Grow and prune a Linear Tree from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            The target values (class labels in classification, real numbers in
            regression).

        weights : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        n_sample, self.n_features_in_ = X.shape

        # check if base_estimator supports fitting with sample_weights
        support_sample_weight = has_fit_parameter(self.base_estimator,
                                                  "sample_weight")

        queue = ['']  # queue of the nodes to evaluate for splitting
        # store the results of each node in dicts
        self._nodes = {}
        self._leaves = {}

        # initialize first fit
        largs = {'classes': None}
        model = deepcopy(self.base_estimator)
        if weights is None or not support_sample_weight:
            model.fit(X[:, self._linear_features], y)
        else:
            model.fit(X[:, self._linear_features], y, sample_weight=weights)

        if hasattr(self, 'classes_'):
            largs['classes'] = self.classes_

        yh = model.predict(X[:, self._linear_features])

        loss = self.loss_func(
            y, yh,
            weights=weights, **largs)

        # Sum loss over targets
        loss = torch.sum(loss)

        self._nodes[''] = Node(
            id=0,
            n_samples=n_sample,
            model=model,
            loss=loss,
            classes=largs['classes']
        )

        # in the beginning consider all the samples
        start = torch.tensor([True], device = X.device).repeat(n_sample)
        mask = start.clone().type(torch.bool)

        possible_mins = []

        if self.min_samples_split is not None:
            if self.min_samples_leaf < 1:
                possible_mins.append(2/self.min_samples_leaf)
            else:
                possible_mins.append(2*len(X)/self.min_samples_leaf)

        if self.min_samples_leaf is not None:
            if self.min_samples_split < 1:
                possible_mins.append(1/self.min_samples_split)
            else:
                possible_mins.append(len(X)/self.min_samples_split)

        possible_mins.append(2**(self.max_depth+1))

        estimated_leaves = min(possible_mins)

        if not hasattr(self, 'disable_tqdm'):
            self.disable_tqdm = False

        with tqdm(total = estimated_leaves, disable = self.disable_tqdm) as pbar:
            pbar.set_postfix_str('Progress is estimated. Tree may finish training normally at any point beyond 50% progress.')
            i = 1

            if self.depth_first:
                active_index = -1
            else:
                active_index = 0

            while len(queue) > 0:
                pbar.update(1)

                if torch.sum(mask) < 2 * self._min_samples_leaf:
                    split_t, split_col, left_node, right_node = None, None, None, None

                elif (self.max_batch_size != torch.inf) and (torch.sum(mask) > self.max_batch_size):
                    valid_min_samples = self.max_batch_size * (self._min_samples_leaf / len(X))
                    random_mask = torch.randint(high = len(X[mask]), size = (self.max_batch_size,))
                    if weights is None:
                        split_t, split_col, left_node, right_node = self._split(
                            X[mask][random_mask], y[mask][random_mask],
                            loss=loss, min_samples_leaf = valid_min_samples)
                    else:
                        split_t, split_col, left_node, right_node = self._split(
                            X[mask][random_mask], y[mask][random_mask], weights[mask][random_mask],
                            loss=loss, min_samples_leaf = valid_min_samples)

                else:
                    if weights is None:
                        split_t, split_col, left_node, right_node = self._split(
                            X[mask], y[mask],
                            loss=loss, min_samples_leaf = self._min_samples_leaf)
                    else:
                        split_t, split_col, left_node, right_node = self._split(
                            X[mask], y[mask], weights[mask],
                            loss=loss, min_samples_leaf = self._min_samples_leaf)

                # no utility in splitting
                if split_col is None or len(queue[active_index]) >= self.max_depth:
                    self._leaves[queue[active_index]] = self._nodes[queue[active_index]]
                    del self._nodes[queue[active_index]]
                    queue.pop(active_index)

                else:
                    model_left, loss_left, wloss_left, n_left, class_left = \
                        left_node
                    model_right, loss_right, wloss_right, n_right, class_right = \
                        right_node

                    self._nodes[queue[active_index] + 'L'] = Node(
                        id=i, parent=queue[active_index],
                        model=model_left,
                        loss=loss_left,
                        w_loss=wloss_left,
                        n_samples=n_left,
                        threshold=self._nodes[queue[active_index]].threshold[:] + [
                            (split_col, 'L', split_t)
                        ]
                    )

                    self._nodes[queue[active_index] + 'R'] = Node(
                        id=i + 1, parent=queue[active_index],
                        model=model_right,
                        loss=loss_right,
                        w_loss=wloss_right,
                        n_samples=n_right,
                        threshold=self._nodes[queue[active_index]].threshold[:] + [
                            (split_col, 'R', split_t)
                        ]
                    )

                    if hasattr(self, 'classes_'):
                        self._nodes[queue[active_index] + 'L'].classes = class_left
                        self._nodes[queue[active_index] + 'R'].classes = class_right

                    self._nodes[queue[active_index]].children = (queue[active_index] + 'L', queue[active_index] + 'R')

                    i += 2
                    q = queue[active_index]
                    queue.pop(active_index)
                    if len(q) < self.max_depth:
                        queue.extend([q + 'R', q + 'L'])

                if len(queue) > 0:
                    loss = self._nodes[queue[active_index]].loss
                    mask = _predict_branch(X, self._nodes[queue[active_index]].threshold, start.clone())
                    

        pbar.close()
        self.node_count = i

        if self.node_count < estimated_leaves / 2:
            warnings.warn(
                "Training appears to have ended early." +
                "If the fit is insufficiently accurate, try increasing max_bins or max_weight."
            )

        return self

    def _fit(self, X, y, sample_weight=None, check_input = None, missing_values_in_feature_mask = None):
        """Build a Linear Tree of a linear estimator from the training
        set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or also (n_samples, n_targets) for
            multitarget regression.
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        n_sample, n_feat = X.shape

        # Ensure y is 2-dimensional
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if isinstance(self.min_samples_split, numbers.Integral):
            if self.min_samples_split < 6:
                raise ValueError(
                    "min_samples_split must be an integer greater than 5 or "
                    "a float in (0.0, 1.0); got the integer {}".format(
                        self.min_samples_split))
            self._min_samples_split = self.min_samples_split
        else:
            if not 0. < self.min_samples_split < 1.:
                raise ValueError(
                    "min_samples_split must be an integer greater than 5 or "
                    "a float in (0.0, 1.0); got the float {}".format(
                        self.min_samples_split))

            self._min_samples_split = int(torch.ceil(torch.tensor([self.min_samples_split * n_sample], device = X.device)))

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if self.min_samples_leaf < 3:
                raise ValueError(
                    "min_samples_leaf must be an integer greater than 2 or "
                    "a float in (0.0, 1.0); got the integer {}".format(
                        self.min_samples_leaf))
            self._min_samples_leaf = self.min_samples_leaf
        else:
            if not 0. < self.min_samples_leaf < 1.:
                raise ValueError(
                    "min_samples_leaf must be an integer greater than 2 or "
                    "a float in (0.0, 1.0); got the float {}".format(
                        self.min_samples_leaf))

            self._min_samples_leaf = torch.ceil(torch.tensor([self.min_samples_leaf * n_sample], device = X.device)).type(torch.int)
            self._min_samples_leaf = torch.max(torch.tensor([3], device=X.device), self._min_samples_leaf)

        if not 1 <= self.max_depth:
            raise ValueError("max_depth must be an integer greater than or equal to 1.")

        if not 3 <= self.max_bins:
            raise ValueError("max_bins must be an integer greater than or equal to 3.")

        if self.categorical_features is not None:
            cat_features = torch.unique(self.categorical_features)

            if (cat_features < 0).any() or (cat_features >= n_feat).any():
                raise ValueError(
                    'Categorical features must be in [0, {}].'.format(
                        n_feat - 1))

            if len(cat_features) == n_feat:
                raise ValueError(
                    "Only categorical features detected. "
                    "No features available for fitting.")
        else:
            cat_features = []
        cat_features = torch.Tensor(cat_features).to(X.device)
        self._categorical_features = cat_features

        if self.split_features is not None: 
            split_features = torch.unique(self.split_features)

            if (split_features < 0).any() or (split_features >= n_feat).any():
                raise ValueError(
                    'Splitting features must be in [0, {}].'.format(
                        n_feat - 1))
        else:
            split_features = torch.arange(n_feat, device = X.device)

        if hasattr(self, 'quadratic_features') and self.quadratic_features:
            assert split_features[0] == 0
            assert torch.all(split_features[1:] - split_features[:-1] == 1)
            split_features = torch.arange(n_feat, device = X.device)

        self._split_features = split_features

        if self.linear_features is not None:
            assert type(self.linear_features) == torch.Tensor

            linear_features = torch.unique(self.linear_features)

            if (linear_features < 0).any() or (linear_features >= n_feat).any():
                raise ValueError(
                    'Linear features must be in [0, {}].'.format(
                        n_feat - 1))

        else:
            combined = torch.cat((torch.arange(n_feat, device = X.device), cat_features))
            uniques, counts = combined.unique(return_counts=True)
            linear_features = uniques[counts == 1].type(torch.int)

        if hasattr(self, 'quadratic_features') and self.quadratic_features:
            assert linear_features[0] == 0
            assert torch.all(linear_features[1:] - linear_features[:-1] == 1)
            num_quadratic_features = int(len(linear_features) * (len(linear_features) + 3) / 2)
            linear_features = torch.arange(num_quadratic_features, device = X.device)

        self._linear_features = linear_features.clone().detach().to(X.device)

        with torch.no_grad():
            self._grow(X, y, sample_weight)

        # Fit TorchLinearRegression if the parameters copied from the splitting method are not enough
        # For example, if UQ parameters are needed 

        if self.save_linear_propogation_uncertainty_parameters or self.save_quadratic_uncertainty_parameters:
            X_leaves = torch.zeros((X.shape[0],), dtype=torch.int, device = X.device)

            for L in self._leaves.values():
                mask = _predict_branch(X, L.threshold)
                X_leaves[mask] = L.id

            leaves = list(self._leaves.keys())

            for i in range(len(self._leaves)):
                node_id = self._leaves[leaves[i]].id
                self._leaves[leaves[i]].model.fit(
                    X[X_leaves==node_id][:, self._linear_features],
                    y[X_leaves==node_id],
                    save_linear_propogation_uncertainty_parameters = self.save_linear_propogation_uncertainty_parameters,
                    save_quadratic_uncertainty_parameters = self.save_quadratic_uncertainty_parameters,

                    )

        return self

    @property
    def num_leaves(self):
        return len(self._leaves)

    def __len__(self):
        return self.num_leaves

    def summary(self, feature_names=None, only_leaves=False, max_depth=None):
        """Return a summary of nodes created from model fitting.

        Parameters
        ----------
        feature_names : array-like of shape (n_features, ), default=None
            Names of each of the features. If None, generic names
            will be used (“X[0]”, “X[1]”, …).

        only_leaves : bool, default=False
            Store only information of leaf nodes.

        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree
            is fully generated.

        Returns
        -------
        summary : nested dict
            The keys are the integer map of each node.
            The values are dicts containing information for that node:

                - 'col' (^): column used for splitting;
                - 'th' (^): threshold value used for splitting in the
                  selected column;
                - 'loss': loss computed at node level. Weighted sum of
                  children' losses if it is a splitting node;
                - 'samples': number of samples in the node. Sum of children'
                  samples if it is a split node;
                - 'children' (^): integer mapping of possible children nodes;
                - 'models': fitted linear models built in each split.
                  Single model if it is leaf node;

            (^): Only for split nodes.
            (^^): Only for leaf nodes.
        """
        check_is_fitted(self, attributes='_nodes')

        if max_depth is None:
            max_depth = 20
        if max_depth < 1:
            raise ValueError(
                "max_depth must be > 0, got {}".format(max_depth))

        summary = {}

        if len(self._nodes) > 0 and not only_leaves:

            if (feature_names is not None and
                    len(feature_names) != self.n_features_in_):
                raise ValueError(
                    "feature_names must contain {} elements, got {}".format(
                        self.n_features_in_, len(feature_names)))

            if feature_names is None:
                feature_names = torch.arange(self.n_features_in_)

            for n, N in self._nodes.items():


                cl, cr = N.children
                Cl = (self._nodes[cl] if cl in self._nodes
                      else self._leaves[cl])
                Cr = (self._nodes[cr] if cr in self._nodes
                      else self._leaves[cr])

                summary[N.id] = {
                    'col': feature_names[Cl.threshold[-1][0]].item(),
                    'th': Cl.threshold[-1][-1], #torch.round(Cl.threshold[-1][-1], decimals=5),
                    'loss': Cl.w_loss + Cr.w_loss,
                    'samples': Cl.n_samples + Cr.n_samples,
                    'children': (Cl.id, Cr.id),
                    'models': (Cl.model, Cr.model)
                }

        for l, L in self._leaves.items():


            summary[L.id] = {
                'loss': L.loss, #torch.round(L.loss, decimals=5),
                'samples': L.n_samples,
                'models': L.model
            }

            if hasattr(self, 'classes_'):
                summary[L.id]['classes'] = L.classes

        return summary

    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, )
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; n_nodes)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self, attributes='_nodes')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_,
            cast_to_ndarray = False,
        )

        X_leaves = torch.zeros((X.shape[0],), dtype=torch.int, device = X.device)

        for L in self._leaves.values():
            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue
            X_leaves[mask] = L.id

        return X_leaves

    def decision_path(self, X):
        """Return the decision path in the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        check_is_fitted(self, attributes='_nodes')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_,
            cast_to_ndarray = False,
        )

        indicator = torch.zeros((X.shape[0], self.node_count), dtype='int64')

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            n = L.id
            p = L.parent
            paths_id = [n]

            while p is not None:
                n = self._nodes[p].id
                p = self._nodes[p].parent
                paths_id.append(n)

            indicator[torch.ix_(mask, paths_id)] = 1

        return sp.csr_matrix(indicator)

    def model_to_dot(self, feature_names=None, max_depth=None):
        """Convert a fitted Linear Tree model to dot format.
        It results in ModuleNotFoundError if graphviz or pydot are not available.
        When installing graphviz make sure to add it to the system path.

        Parameters
        ----------
        feature_names : array-like of shape (n_features, ), default=None
            Names of each of the features. If None, generic names
            will be used (“X[0]”, “X[1]”, …).

        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree
            is fully generated.

        Returns
        -------
        graph : pydot.Dot instance
            Return an instance representing the Linear Tree. Splitting nodes have
            a rectangular shape while leaf nodes have a circular one.
        """
        import pydot

        summary = self.summary(feature_names=feature_names, max_depth=max_depth)
        graph = pydot.Dot('linear_tree', graph_type='graph')

        # create nodes
        for n in summary:
            if 'col' in summary[n]:
                if isinstance(summary[n]['col'], str):
                    msg = "id_node: {}\n{} <= {:.4e}\nloss: {:.4e}\nsamples: {}"
                else:
                    msg = "id_node: {}\nX[{}] <= {:.4e}\nloss: {:.4e}\nsamples: {}"

                msg = msg.format(
                    n, summary[n]['col'], float(summary[n]['th']),
                    summary[n]['loss'], summary[n]['samples']
                )
                graph.add_node(pydot.Node(n, label=msg, shape='rectangle'))

                for c in summary[n]['children']:
                    if c not in summary:
                        graph.add_node(pydot.Node(c, label="...",
                                                  shape='rectangle'))

            else:
                msg = "id_node: {}\nloss: {:.4e}\nsamples: {}".format(
                    n, summary[n]['loss'], summary[n]['samples'])
                graph.add_node(pydot.Node(n, label=msg))

        # add edges
        for n in summary:
            if 'children' in summary[n]:
                for c in summary[n]['children']:
                    graph.add_edge(pydot.Edge(n, c))

        return graph

    def plot_model(self, feature_names=None, max_depth=None, format = "png"):
        """Convert a fitted Linear Tree model to dot format and display it.
        It results in ModuleNotFoundError if graphviz or pydot are not available.
        When installing graphviz make sure to add it to the system path.

        Parameters
        ----------
        feature_names : array-like of shape (n_features, ), default=None
            Names of each of the features. If None, generic names
            will be used (“X[0]”, “X[1]”, …).

        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree
            is fully generated.

        Returns
        -------
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
        Splitting nodes have a rectangular shape while leaf nodes
        have a circular one.
        """
        from IPython.display import Image
        from IPython.display import SVG

        graph = self.model_to_dot(feature_names=feature_names, max_depth=max_depth)
        if format == "png":
            return Image(graph.create_png())
        elif format == "svg":
            return SVG(graph.create_svg())
        else:
            raise NotImplementedError("Unsupported Format")


class _LinearForest(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        base_estimator=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super(RandomForestRegressor, self).__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.base_estimator = base_estimator

        self._fit = super().fit


def tree_from_json(filename, torch_device = 'cpu'):
    """Load a tree from a json file.

    Parameters
    ----------
    filename : str
        The path to the json file.

    Returns
    -------
    tree : dict
        The tree in a dictionary format.
    """

    from hyperplanetree.hyperplane_tree import HyperplaneTreeRegressor

    params = json.load(open(filename, 'r'))
    params['_leaves'] = {}
    params['_nodes'] = {}

    params['base_estimator'] = TorchLinearRegression()

    for key, value in params['nodes'].items():
        if 'model' in value:
            # Node is leaf node
            node = Node(
                id = int(key),
                model = type(params['base_estimator'])(),
                loss = torch.tensor(value['loss'], device=torch_device),
                n_samples = torch.tensor(value['samples'], device=torch_device),
                #w_loss = torch.tensor(value['w_loss'], device=torch_device)
        )
            node.model.params = torch.Tensor(value['model']['params']).to(torch_device)
            node.threshold = None
            params['_leaves'][int(key)] = node

        else:
            # Node is splitting node
            node = Node(
                id = int(key),
                model = type(params['base_estimator'])(),
                loss = torch.tensor(value['loss'], device=torch_device),
                n_samples = torch.tensor(value['samples'], device=torch_device),
                #w_loss = torch.tensor(value['w_loss'], device=torch_device),
                children = (int(value['children'][0]), int(value['children'][1])),
        )
            node.col = value['col']
            node.th = value['th']
            params['_nodes'][int(key)] = node

    for id, node in params['_nodes'].items():
        if id == '0':
            node.threshold = []

        cur_thresh = node.threshold
        new_row_L = (
            torch.tensor(node.col).to(torch_device),
            'L',
            torch.tensor(node.th).to(torch_device),
        )
        
        new_thresh_L = cur_thresh + [new_row_L]

        if node.children[0] in params['_nodes'].keys():
            params['_nodes'][node.children[0]].threshold = new_thresh_L
        elif node.children[0] in params['_leaves'].keys():
            params['_leaves'][node.children[0]].threshold = new_thresh_L

        new_row_R = (
            torch.tensor(node.col).to(torch_device),
            'R',
            torch.tensor(node.th).to(torch_device),
        )
        
        new_thresh_R = cur_thresh + [new_row_R]

        if node.children[1] in params['_nodes'].keys():
            params['_nodes'][node.children[1]].threshold = new_thresh_R
        elif node.children[1] in params['_leaves'].keys():
            params['_leaves'][node.children[1]].threshold = new_thresh_R

    tree = HyperplaneTreeRegressor(
        criterion = params['criterion'],
        max_depth = params['max_depth'],
        min_samples_leaf = params['min_samples_leaf'],
        categorical_features = torch.LongTensor(params['categorical_features']).to(torch_device),
        linear_features = torch.LongTensor(params['linear_features']).to(torch_device),
        split_features = torch.LongTensor(params['split_features']).to(torch_device),
        )
    
    tree.linear_combinations_transform.final_matrix = torch.FloatTensor(params['hyperplanes_final_matrix']).to(torch_device)
        
    tree.n_features_in_ = params['n_features_in']
    tree.n_targets_ = params['n_targets']
    tree._linear_features = torch.LongTensor(params['linear_features']).to(torch_device)

    tree._nodes = params['_nodes']
    tree._leaves = params['_leaves']

    return tree

def plot_surrogate_2d(model, features, cmap = 'rainbow'):
    import matplotlib.pyplot as plt

    leaf = model.apply(features)
    y_pred = model.predict(features)

    fig = plt.figure(figsize=(5,5))
    plt.scatter(features[:, 0], features[:, 1], c=leaf, marker='s', s=4, cmap=cmap)

    plt.xlabel('X[0]', fontsize=12)
    plt.ylabel('X[1]', fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], y_pred.flatten(), marker='.', s=20, c=leaf, cmap=cmap)  
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    plt.show()