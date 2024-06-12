import systems2atoms as s2a
model = s2a.surrogates.model_initializer()
s2a.surrogates.solve(model, solver='glpk')
s2a.surrogates.pprint(model)