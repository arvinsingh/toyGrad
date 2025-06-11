import numpy as np
import matplotlib.pyplot as plt

from core.special import SScalar


def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

def plot_decision_boundary(model, X_data, y_data):
    plt.figure(figsize=(12, 8))
    
    # mesh
    h = 0.02
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    
    for point in mesh_points:
        x_scalar = [SScalar(float(point[0])), 
                   SScalar(float(point[1]))]
        pred = model(x_scalar)
        Z.append(pred.data)
    
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    
    # decision boundary
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # data points
    plt.scatter(X_data[y_data==0, 0], X_data[y_data==0, 1], 
                c='red', marker='o', s=50, edgecolors='black', label='Class 0')
    plt.scatter(X_data[y_data==1, 0], X_data[y_data==1, 1], 
                c='blue', marker='s', s=50, edgecolors='black', label='Class 1')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.colorbar(label='Prediction Probability')
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_model(model, X_scalar, y_scalar):
    correct = 0
    total = len(X_scalar)
    
    predictions = []
    true_labels = []
    
    for x_batch, y_true in zip(X_scalar, y_scalar):
        y_pred = model(x_batch)
        pred_class = 1 if y_pred.data > 0.5 else 0
        true_class = int(y_true.data)
        
        predictions.append(pred_class)
        true_labels.append(true_class)
        
        if pred_class == true_class:
            correct += 1
    
    accuracy = correct / total
    
    # confusion matrix manual calculation
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)

    print(f"Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.4f}")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.4f}")