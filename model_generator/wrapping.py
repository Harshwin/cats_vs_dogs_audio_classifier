
import mlflow

class ModelWrapper(mlflow.pyfunc.PythonModel):
    ## defining objects needed for model prediction.
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    def predict(self, model_input):
        data_features = self.preprocess(model_input)
        return self.model.predict(data_features)

