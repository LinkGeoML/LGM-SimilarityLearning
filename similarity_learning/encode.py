from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional


def lstm1(n_units) -> Sequential:
    """
    Simple encoder with just one LSTM layer.
    Parameters
    ----------
    n_units : int
        Number of units for the LSTM layer.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='single_lstm_encoder')
    model.add(LSTM(units=n_units))
    return model


def lstm2(n_units) -> Sequential:
    """
    Encoder with 2 stacked LSTM Layers.
    Parameters
    ----------
     n_units : int
        Number of units for the LSTM layers.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='stacked_lstm_encoder')
    model.add(LSTM(units=n_units, return_sequences=True))
    model.add(LSTM(units=n_units))
    return model


def bilstm1(n_units) -> Sequential:
    """
    Encoder with 1 Bidirectional LSTM Layer.
    Parameters
    ----------
     n_units : int
        Number of units for the LSTM layer.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='single_bi_lstm_encoder')
    model.add(Bidirectional(LSTM(units=n_units)))
    return model


def bilstm2(n_units) -> Sequential:
    """
    Encoder with 2 Bidirectional LSTM Layers.
    Parameters
    ----------
     n_units : int
        Number of units for the LSTM layers.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='stacked_bi_lstm_encoder')
    model.add(Bidirectional(LSTM(units=n_units, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=n_units)))
    return model


def gru1(n_units) -> Sequential:
    """
    Simple encoder with just one GRU layer.
    Parameters
    ----------
    n_units : int
        Number of units for the GRU layer.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='single_gru_encoder')
    model.add(GRU(units=n_units))
    return model


def gru2(n_units) -> Sequential:
    """
    Encoder with 2 stacked LSTM Layers.
    Parameters
    ----------
     n_units : int
        Number of units for the LSTM layers.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='stacked_gru_encoder')
    model.add(GRU(units=n_units, return_sequences=True))
    model.add(GRU(units=n_units))
    return model


def bigru1(n_units) -> Sequential:
    """
    Encoder with 1 Bidirectional LSTM Layer.
    Parameters
    ----------
     n_units : int
        Number of units for the LSTM layer.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='single_bi_gru_encoder')
    model.add(Bidirectional(GRU(units=n_units)))
    return model


def bigru2(n_units) -> Sequential:
    """
    Encoder with 2 Bidirectional LSTM Layers.
    Parameters
    ----------
     n_units : int
        Number of units for the LSTM layers.

    Returns
    -------
    Sequential
    """
    model = Sequential(name='stacked_bi_gru_encoder')
    model.add(Bidirectional(GRU(units=n_units, return_sequences=True)))
    model.add(Bidirectional(GRU(units=n_units)))
    return model
