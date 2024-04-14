from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure
model = BayesianModel([('Cough', 'Flu'), ('Fever', 'Flu')])

# Define CPDs
cpd_cough = TabularCPD('Cough', 2, [[0.5], [0.5]])
cpd_fever = TabularCPD('Fever', 2, [[0.3], [0.7]])
cpd_flu = TabularCPD('Flu', 2, [[0.95, 0.1, 0.8, 0.3], [0.05, 0.9, 0.2, 0.7]],
                      evidence=['Cough', 'Fever'], evidence_card=[2, 2])

# Add CPDs
model.add_cpds(cpd_cough, cpd_fever, cpd_flu)

# Perform inference
inference = VariableElimination(model)
posterior = inference.query(variables=['Flu'], evidence={'Cough': 1, 'Fever': 1})

print(posterior)
