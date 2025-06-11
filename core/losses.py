

class MSELoss:
    def __call__(self, y_pred, y_true):
        # Mean Squared Error: 1/n * sum((y_pred - y_true)^2)
        loss = (y_pred - y_true) ** 2
        return loss

class BCELoss:
    def __call__(self, y_pred, y_true):
        # Binary Cross-Entropy: -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        loss = -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
        return loss

class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        # Cross-Entropy Loss: -sum(y_true * log(y_pred))
        loss = -(y_true * y_pred.log())
        return loss