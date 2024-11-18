import torch
import itertools

from sklearn.base import TransformerMixin, BaseEstimator

from ._classes import recursive_to_device

class LinearCombinations(TransformerMixin, BaseEstimator):
    """SKLearn transformer for finite number of linear combinations

    An SKLearn transformer for feature engineering that returns a finite
    number of linear combinations of the input features.

    Parameters
    ----------
    LCs : array_like or None
        List of all linear combination weights

    num_terms: int or None
        Number of terms in each linear combination
        If None, determined automatically from shape of LCs
        Currently, only 2 is supported

    do_symmetrize: bool (default: True)
        Whether or not to symmetrize the input LCs. e.g. if you enter [[1, 2]]
        you will get [[1, -2], [1, -0.5], [1, 0.5], [1,2]].

    do_scaling: bool (default: True)
        Automatically scale weights in LCs so that the range of each feature is the same

    tol_decimals: int (default: 4)
        How many decimal places of precision. Useful when reducing degeneracies
        after symmetrization step.

    torch_device: torch.device or str or None (default: None)
        Device where tensors are created. If None, will default to LCs.device
        If LCs is also None, will default to 'cpu'

    max_weight: int
        If generating LCs in this function, highest LC weight to consider
    """
    def __init__(self,
                 LCs = None,
                 num_terms = None,
                 do_symmetrize = True,
                 do_scaling = True,
                 tol_decimals = 4,
                 torch_device = None,
                 max_weight = None,
                 categorical_features = None,
                 ):
        
        if LCs is not None:
            assert type(LCs) == torch.Tensor
        
        # Set torch device
        if torch_device is None and LCs is not None:
            torch_device = LCs.device
        elif torch_device is None and LCs is None:
            torch_device = 'cpu'

        if LCs is None and num_terms is None:
            num_terms = 2

        if LCs is None:
            if max_weight is None:
                max_weight = 1

            if max_weight == 0:
                LCs = torch.eye(num_terms, device = torch_device)

            else:
                LCs = generate_planes_to_index(dimension = num_terms, max_weight = max_weight, device = torch_device)

        if num_terms is None:
            num_terms = len(LCs[0])

        assert len(LCs[0]) == num_terms

        if do_symmetrize:
            LCs = symmetrize(
                LCs = LCs,
                tol_decimals = tol_decimals,
            )

        self.final_matrix = None

        self.LCs = LCs
        self.do_symmetrize = do_symmetrize
        self.num_terms = num_terms
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.max_weight = max_weight
        self.do_scaling = do_scaling
        self.categorical_features = categorical_features

        if self.categorical_features is not None:
            self.categorical_features = torch.unique(self.categorical_features)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            
        if (self.final_matrix is None) or (X.shape[1] != len(self.final_matrix)):
            # move LCs to correct device
            self.LCs = self.LCs.to(X.device)

            # Remove identity rows from LCs
            self.LCs = self.LCs[torch.sum(torch.abs(self.LCs), dim=1) != 1]

            num_cols = X.shape[1]
            if self.categorical_features is not None:
                num_cols_non_categorical = num_cols - len(self.categorical_features)
                cols_to_perm = [col for col in range(num_cols) if col not in self.categorical_features]
            else:
                num_cols_non_categorical = num_cols
                cols_to_perm = range(num_cols)

            num_out_cols = int(num_cols + torch.math.factorial(num_cols_non_categorical) / \
                (torch.math.factorial(num_cols_non_categorical - self.num_terms)) * len(self.LCs))

            perms = itertools.permutations(cols_to_perm, self.num_terms)
            final_matrix = torch.zeros((num_out_cols, num_cols), device = X.device)
            final_matrix[:num_cols] = torch.eye(num_cols)

            for i, indices in enumerate(perms):
                for j, LC in enumerate(self.LCs):
                    final_matrix[num_cols + i*len(self.LCs) + j, indices] = LC


            # find first nonzero entry in each row and normalize
            leftmost_nonzero = torch.argmax((final_matrix != 0).type(torch.int), axis=1)
            leftmost_nonzero = final_matrix[range(len(final_matrix)), leftmost_nonzero]
            final_matrix = (final_matrix.T / leftmost_nonzero).T

            # take only unique rows in final matrix
            final_matrix = torch.unique(
                torch.round(
                    final_matrix[num_cols:].to('cpu'),
                    decimals=self.tol_decimals,
                    ),
                dim=0,
                sorted=True,
                ).to(X.device)
            
            if self.do_scaling:
                ranges = torch.max(X, dim=0)[0] - torch.min(X, dim=0)[0]
                epsilon = 1e-8  # Small value to avoid division by zero
                ranges[ranges < epsilon] = epsilon
                final_matrix /= ranges

                # Renormalize
                leftmost_nonzero = torch.argmax((final_matrix != 0).type(torch.int), axis=1)
                leftmost_nonzero = final_matrix[range(len(final_matrix)), leftmost_nonzero]
                final_matrix = (final_matrix.T / leftmost_nonzero).T
            
            # Insert identity matrix to first rows
            self.final_matrix = torch.zeros((num_cols+len(final_matrix), num_cols), device = X.device)
            self.final_matrix[:num_cols] = torch.eye(num_cols, device = X.device)
            self.final_matrix[num_cols:] = final_matrix

            self.final_matrix = self.final_matrix.T.type(X.dtype)
        return X @ self.final_matrix
    
    def to(self, device):
        return recursive_to_device(self, device)

    
def generate_angular_lcs_2d(
        divisions: int,
        device = 'cpu'
        ):
    """Generate lines in two dimensions with equal angle spacing

    Parameters
    ----------
        divisions : int, 1 or greater
            How many lines in each quadrant

        device : torch.device or str
            device where the tensor will be created

    Returns
    -------
    spacing : torch.Tensor 
        intended to be used as hyperplane_weights when initializing a tree
    
    """
    assert divisions >= 1

    spacing = torch.Tensor([[torch.sin(x), torch.cos(x)] for x in torch.linspace(0, torch.pi/2, divisions+2)])[1:-1]
    spacing = (spacing.T / spacing[:,0]).T.to(device)
    return spacing

def generate_planes_to_index(
        dimension: int, 
        max_weight: int=3,
        device = 'cpu',
        tol_decimals: int=4
        ):
    """Generate hyperplanes based on Miller index-like system

    Generates all possible planes with integer weights up to
    and including the specified max.
    Automatically reduces degenerate weight combinations.
    Automatically normalizes so the highest magnitude weight is 1.
    Does not produce negative weights.
    It is highly recommended to use this with "symmetrize" set to True
    in your tree initialization arguments to obtain all symmetries.

    Parameters
    ----------
    dimension : int
        How many terms in the resulting planes

    max_weight : int
        Highest possible weight in the generated planes

    Returns
    -------
    out : torch.Tensor 
        intended to be used as hyperplane_weights when initializing a tree

    Example
    -------
    dimension = 2, max_index = 3 ==>

    [
        [1.0000, 0.0000], # (1, 0) plane
        [1.0000, 0.3333], # (3, 1) plane
        [1.0000, 0.5000], # (2, 1) plane
        [1.0000, 0.6667], # (3, 2) plane
        [1.0000, 1.0000], # (1, 1) plane
    ]
    
    """

    out = itertools.combinations_with_replacement(range(max_weight, -1 , -1), dimension)
    out = torch.Tensor(list(out))[:-1]
    out = (out.T / out[:, 0]).T
    out.round(decimals = tol_decimals)
    out = torch.unique(out, dim = 0).to(device)
    return out

def symmetrize(
    LCs,
    tol_decimals = 4,
):
    """Symettrize linear combination weights
    
    Helper function for LinearCombinations transform.
    Symmetrizes provided linear combination weights with respect to parity and permutations.

    Parameters
    ----------
    LCs : tensor
        Tensor of linear combination weights
    
    tol_decimals : int
        Number of decimal places to consider when reducing non-unique LC weights

    Returns
    -------
    LCs : tensor
        Tensor of linear combination weights
    """

    num_terms = len(LCs[0])
    torch_device = LCs.device

    # Symmetrize +/- parity
    parity_matrix = torch.Tensor(tuple(itertools.product([1, -1], repeat = num_terms))).to(torch_device)
    parity_matrix = parity_matrix[:, None, :]
    LCs = torch.reshape(LCs * parity_matrix, (-1, num_terms))

    # Symmetrize permutations
    permutations_matrix = torch.Tensor(tuple((itertools.permutations(range(num_terms))))).to(torch_device).type(torch.int)
    LCs = torch.reshape(LCs[:, permutations_matrix], (-1, num_terms))

    # Remove LCs with non-trailing zeros
    previous_was_zero = torch.zeros(len(LCs), dtype=bool, device = torch_device)
    keep = torch.ones(len(LCs), dtype=bool, device = torch_device)
    for i in range(num_terms):
        keep = torch.logical_and(torch.logical_not(torch.logical_and(previous_was_zero, LCs[:, i] != 0)), keep)
        previous_was_zero = LCs[:, i] == 0

    LCs = LCs[keep]

    # Normalize all combinations
    LCs = (LCs.T / LCs[:, 0]).T

    # Only take unique LCs
    LCs = torch.unique(torch.round(LCs.to('cpu'), decimals=tol_decimals), dim=0).to(torch_device)

    return LCs
