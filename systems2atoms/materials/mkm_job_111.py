from catmap import ReactionModel

mkm_file = 'HCOOH_decomposition_111.mkm'
model = ReactionModel(setup_file=mkm_file)
model.output_variables += ['production_rate', 'consumption_rate', 'turnover_frequency']
model.run()

#M = mean_field_solver.MeanFieldSolver(model)
#print(M.get_rate())

from catmap import analyze
vm = analyze.VectorMap(model)
vm.plot_variable = 'production_rate' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-25 #minimum rate to plot
vm.max = 1e8 #maximum rate to plot
vm.threshold = 1e-25 #anything below this is considered to be 0
vm.subplots_adjust_kwargs = {'left':0.2,'right':0.8,'bottom':0.15}
vm.plot(save='production_rate.pdf')

vm.plot_variable = 'rate' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-25 #minimum rate to plot
vm.max = 1e3 #maximum rate to plot
vm.plot(save='rate.pdf') #draw the plot and save it as "rate.pdf"

vm.plot_variable = 'coverage' #tell the model which output to plot
vm.log_scale = False #rates should be plotted on a log-scale
vm.min = 0 #minimum rate to plot
vm.max = 1 #maximum rate to plot
vm.plot(save='coverage.pdf') #draw the plot and save it as "rate.pdf"


vm.plot_variable = 'consumption_rate' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-25 #minimum rate to plot
vm.max = 1e8 #maximum rate to plot
vm.threshold = 1e-25 #anything below this is considered to be 0
#vm.subplots_adjust_kwargs = {'left':0.2,'right':0.8,'bottom':0.15}
vm.plot(save='consumption_rate.pdf')

vm.plot_variable = 'turnover_frequency' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-25 #minimum rate to plot
vm.max = 1e8 #maximum rate to plot
#vm.threshold = 1e-25 #anything below this is considered to be 0
#vm.subplots_adjust_kwargs = {'left':0.2,'right':0.8,'bottom':0.15}
vm.plot(save='turnover_frequency.pdf')
