from gaussian_model import GaussianGenerativeModel
from evaluation import accuracy

def tune_lambda(X_train, y_train, X_val, y_val):
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    best_lambda = None
    best_acc = -1
    val_accuracies = []

    print("Lambda\tValidation Accuracy")
    print("---------------------------")
    for lambdaa in lambdas:
        model = GaussianGenerativeModel(lambda_reg=lambdaa)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy(y_val, preds)
        val_accuracies.append(acc)
        print(f"{lambdaa:.4f}\t{acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_lambda = lambdaa

    print("---------------------------")
    print(f"Best λ: {best_lambda:.4f}, Validation Accuracy: {best_acc:.4f}\n")
    return best_lambda, best_acc  
