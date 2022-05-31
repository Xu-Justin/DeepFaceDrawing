from sklearn.preprocessing import MinMaxScaler

def tanh(x):
    return (x * 2) - 1

def itanh(x):
    return (x + 1) / 2

def rescale(numpy):
    scaler = MinMaxScaler()
    return scaler.fit_transform(numpy)