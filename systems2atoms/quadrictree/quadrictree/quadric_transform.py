import torch
import itertools
import copy

from sklearn.base import TransformerMixin, BaseEstimator

from ._classes import recursive_to_device


class QuadricTransform(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        max_weight = 2,
        num_terms = 2,
        torch_device = 'cpu',
        categorical_features = None,
        tol_decimals = 4,
        do_scaling = True,
        normalize_range = 2,
    ):

        self.max_weight = max_weight
        self.num_terms = num_terms
        self.torch_device = torch_device
        self.categorical_features = categorical_features
        self.tol_decimals = tol_decimals
        self.do_scaling = do_scaling
        self.normalize_range = normalize_range

        self.quadric_matrix = None

    def fit(self, X, y=None):
        return self

    def generate_combinations_with_parity(self, X):
        def generate_combinations(current_comb, non_zero_count, num_features):
            if len(current_comb) == num_features:
                if non_zero_count <= self.num_terms and non_zero_count > 0:
                    all_combinations.append(current_comb)
                return
            
            for weight in range(-self.max_weight, self.max_weight + 1):
                new_non_zero_count = non_zero_count + (1 if weight != 0 else 0)
                if new_non_zero_count <= self.num_terms:
                    generate_combinations(current_comb + [weight], new_non_zero_count, num_features)

        all_combinations = []
        num_features = int((X.shape[-1] ** 2 + X.shape[-1]) / 2) - 1
        generate_combinations([], 0, num_features)
        
        # Convert to torch.Tensor
        all_combinations = torch.tensor(all_combinations, dtype=X.dtype).to(X.device)
        
        # Normalize and keep non-degenerate
        leftmost_nonzero = torch.argmax((all_combinations != 0).type(torch.int), axis=1)
        leftmost_nonzero_values = all_combinations[range(len(all_combinations)), leftmost_nonzero]
        normalized_combinations = (all_combinations.T / leftmost_nonzero_values).T
        
        # Take only unique rows
        unique_combinations = torch.unique(
            torch.round(
                normalized_combinations.to('cpu'),
                decimals=self.tol_decimals,
            ),
            dim=0,
            sorted=True,
        ).to(X.device)
        
        # sort
        row_sums = torch.sum(torch.abs(unique_combinations), dim=1)
        sorted_indices = torch.argsort(row_sums)
        sorted_permutations = unique_combinations[sorted_indices]
        length = sorted_permutations.shape[-1]

        sorted_permutations[:length, :length] = torch.eye(length, device=unique_combinations.device, dtype = torch.float)

        return sorted_permutations

    def build_quadric_matrix(self, X):
        # Generate permutations of weights
        all_permutations = self.generate_combinations_with_parity(X)

        # Convert all_permutations to matrix
        quadric_matrix = torch.zeros(
            (len(all_permutations), X.shape[-1], X.shape[-1]),
            device = X.device,
            dtype = torch.float
        )

        mask = torch.triu(torch.ones_like(quadric_matrix)).type(torch.bool)
        mask[:, 0, 0] = False

        quadric_matrix[mask] = all_permutations.reshape(-1)

        return quadric_matrix

    def normalizer(self, X):
        if not hasattr(self, 'normalizer_weight'):
            self.normalizer_weight = self.normalize_range/(torch.max(X, dim = -2)[0] - torch.min(X, dim = -2)[0])
            self.normalizer_bias = self.normalize_range/2*(torch.max(X, dim = -2)[0] + torch.min(X, dim = -2)[0])/(torch.max(X, dim = -2)[0] - torch.min(X, dim = -2)[0])


            self.normalizer_weight[0] = 1
            self.normalizer_bias[0] = 0

        return (X * self.normalizer_weight) - self.normalizer_bias


    def transform(self, X):
        if self.categorical_features is not None:
            categorical_mask = torch.zeros_like(X[0]).type(torch.bool)
            categorical_mask[self.categorical_features] = True
            X_categorical = copy.deepcopy(X[:, categorical_mask])
            X = X[:, ~categorical_mask]

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        X = torch.hstack((torch.ones((len(X),1), device = X.device), X))

        if self.do_scaling:
            X = self.normalizer(X)

        if (self.quadric_matrix is None) or X.shape[1] != len(self.quadric_matrix):
           self.quadric_matrix = self.build_quadric_matrix(X)

        X = torch.einsum('fs,qfg,sg->sq', X.T, self.quadric_matrix,X)

        if self.categorical_features is not None:
            X_out = torch.zeros(X.shape[0], X.shape[1]+X_categorical.shape[1], dtype=torch.float, device=X.device)

            categorical_mask = torch.zeros_like(X_out[0]).type(torch.bool)
            categorical_mask[self.categorical_features] = True

            X_out[:, categorical_mask] = X_categorical
            X_out[:, ~categorical_mask] = X
            X = copy.deepcopy(X_out)

        return X