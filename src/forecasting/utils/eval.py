from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    RMSE = root_mean_squared_error(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return RMSE, MAE, r2