import systems2atoms as s2a

def test_surrogates():
    model = s2a.surrogates.model_initializer()
    s2a.surrogates.pprint(model)