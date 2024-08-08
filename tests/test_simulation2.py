import pandas as pd

class TestMonthlyRetraining:
    def generate_simulated_data(self, week):
        # Simplified function for testing
        simulated_data = pd.DataFrame({
            'step': [1],
            'type': ['PAYMENT'],
            'amount': [100.0],
            'oldbalanceOrg': [200.0],
            'newbalanceOrig': [300.0],
            'nameOrig': ['name'],
            'nameDest': ['dest'],
            'oldbalanceDest': [0.0],
            'newbalanceDest': [0.0]
        })
        simulated_data['week'] = week
        return simulated_data

    def test_function_call(self):
        # Call the function directly
        simulated_data = self.generate_simulated_data(1)
        print(simulated_data)

# Run the test
test = TestMonthlyRetraining()
test.test_function_call()
