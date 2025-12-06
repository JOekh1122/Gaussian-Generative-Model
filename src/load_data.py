from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # hena insert data
    digits = load_digits()
    X = digits.data
    y = digits.target

    #hena ta2semt data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    # hena sacel data
    scaler = StandardScaler()
    X_train_sacled = scaler.fit_transform(X_train)
    X_val_sacled  = scaler.transform(X_val)
    X_test_sacled = scaler.transform(X_test)

    return X_train_sacled , X_val_sacled ,X_test_sacled , y_train, y_val, y_test
