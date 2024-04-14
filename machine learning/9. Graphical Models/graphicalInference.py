# Code for learning and inference in graphical models for predictive maintenance
# would involve data preprocessing, structure learning, parameter estimation,
# and real-time inference, which can be quite involved.
# Here's a simplified example for learning and inference with known structure.

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure (assume known for simplicity)
model = BayesianModel([('Sensor1', 'Failure'), ('Sensor2', 'Failure')])

# Define CPDs (assume known for simplicity)
cpd_sensor1 = TabularCPD('Sensor1', 2, [[0.8], [0.2]])
cpd_sensor2 = TabularCPD('Sensor2', 2, [[0.9], [0.1]])
cpd_failure = TabularCPD('Failure', 2, [[0.99, 0.05], [0.01, 0.95]],
                         evidence=['Sensor1', 'Sensor2'], evidence_card=[2, 2])

# Add CPDs
model.add_cpds(cpd_sensor1, cpd_sensor2, cpd_failure)

# Perform inference
inference = VariableElimination(model)
posterior = inference.query(variables=['Failure'], evidence={'Sensor1': 1, 'Sensor2': 0})

print(posterior)
