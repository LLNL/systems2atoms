import torch

from systems2atoms.systems2atoms.surrogates import model_definition
torch_device = 'cpu'
from systems2atoms.quadrictree import LinearTreeRegressor, HyperplaneTreeRegressor, QuadricTreeRegressor

def generate_function(x, y, noise_scale = 0.1):
    a = x/1.3 + y   
    f = a*x*y + a**2 + 4*torch.cos(3*a)
    
    f += torch.normal(mean = torch.zeros_like(x), std = noise_scale)

    return f.to(torch_device)

def test_quadrictree():
    # Generate sampling points
    x0 = torch.linspace(-3, 3, 20, device = torch_device)
    x1 = torch.linspace(-3, 3, 20, device = torch_device)
    X0, X1 = torch.meshgrid(x0, x1, indexing='ij')

    # Generate features and labels tensors
    features = torch.vstack((X0.flatten(), X1.flatten())).T.type(torch.float)
    y = generate_function(X0, X1).type(torch.float).flatten()

    shuffle = torch.randperm(len(y))

    train_indices = shuffle[:int(0.8*len(shuffle))]
    test_indices = shuffle[int(0.8*len(shuffle)):]

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_y = y[train_indices]
    test_y = y[test_indices]

    model = LinearTreeRegressor()
    model.fit(train_features, train_y)
    y_pred = model.predict(test_features.to(torch_device))
    leaves = len(model)
    assert leaves > 2
    assert max(y_pred) - min(y_pred) > 0

    model = HyperplaneTreeRegressor()
    model.fit(train_features, train_y)
    y_pred = model.predict(test_features.to(torch_device))
    leaves = len(model)
    assert leaves > 2
    assert max(y_pred) - min(y_pred) > 0

    model = QuadricTreeRegressor()
    model.fit(train_features, train_y)
    y_pred = model.predict(test_features.to(torch_device))
    leaves = len(model)
    assert leaves > 2
    assert max(y_pred) - min(y_pred) > 0