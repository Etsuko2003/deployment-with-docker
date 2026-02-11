import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers, layers, models, callbacks

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(keys="UDI", drop=True)
    df = df.drop(columns=["Product ID"])
    df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"]).copy()
    df = pd.get_dummies(df, columns=["Type"], prefix="Type", drop_first=True, dtype=int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Power_kw"] = (df["Torque [Nm]"] * df["Rotational speed [rpm]"]) / 9550
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Speed_torque_ratio"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1)
    max_wear = df["Tool wear [min]"].max()
    df["Wear_level"] = pd.cut(
        df["Tool wear [min]"],
        bins=[0, 80, 160, max_wear + 1],
        labels=[0, 1, 2],
        include_lowest=True,
    )
    df["Wear_level"] = (
        pd.to_numeric(df["Wear_level"], errors="coerce").fillna(0).astype(int)
    )
    df["High_wear"] = (df["Tool wear [min]"] > 200).astype(int)
    df["Thermal_load"] = df["Temp_diff"] * df["Power_kw"]
    df["Mechanical_stress"] = df["Torque [Nm]"] * (1 + df["Tool wear [min]"] / 250)

    return df


def clean_data(df: pd.DataFrame, correlation_threshold=0.9) -> pd.DataFrame:
    corr = df.drop(columns=["Machine failure"]).corr(method="pearson")
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index()
    )

    corr_pairs.columns = ["var1", "var2", "correlation"]
    strong_corr_pairs = corr_pairs[corr_pairs["correlation"] > correlation_threshold]

    features_to_drop = []
    for _, row in strong_corr_pairs.iterrows():
        var1, var2 = row["var1"], row["var2"]
        if df[var1].var() < df[var2].var():
            features_to_drop.append(var1)
            print("Should drop", var1)
        else:
            features_to_drop.append(var2)
            print("Should drop", var2)

    df = df.drop(columns=features_to_drop)

    return df


def get_X_y(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    return X, y


def get_tvt_data(X, y) -> tuple:
    n_total = len(X)

    n_train = int(0.8 * n_total)
    n_val = int(0.15 * n_total)

    X_train_full = X[:n_train]
    y_train_full = y[:n_train]
    X_train = X_train_full[y_train_full == 0]
    y_train = y_train_full[y_train_full == 0]

    X_val_full = X[n_train : n_train + n_val]
    y_val_full = y[n_train : n_train + n_val]
    X_val = X_val_full[y_val_full == 0]
    y_val = y_val_full[y_val_full == 0]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test


def create_sequences(X, timesteps):
    sequences = []
    for i in range(len(X) - timesteps + 1):
        sequences.append(X[i : i + timesteps])
    return np.array(sequences)


def build_models(timesteps, n_features, latent_dim=32):

    # ========= ENCODER =========
    encoder_inputs = keras.Input(shape=(timesteps, n_features))

    x = layers.LSTM(64, return_sequences=True)(encoder_inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(latent_dim)(x)

    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = keras.Model(encoder_inputs, latent, name="encoder")

    # ========= DECODER =========
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.RepeatVector(timesteps)(latent_inputs)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)

    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # ========= AUTOENCODER =========
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.Model(encoder_inputs, autoencoder_outputs, name="autoencoder")

    autoencoder.compile(optimizer="adam", loss="mse")

    return encoder, decoder, autoencoder


if __name__ == "__main__":
    # Constantes
    CORRELATION_THRESHOLD = 0.92
    TIMESTEPS = 20
    N_EPOCHS = 5
    BATCH_SIZE = 32
    N_PATIENCE = 5

    df = pd.read_csv("datas/ai4i2020.csv")

    df = preprocess_data(df)
    df = build_features(df)
    df = clean_data(df, correlation_threshold=CORRELATION_THRESHOLD)
    X, y = get_X_y(df)
    X_train, X_val, X_test, _, _, y_test = get_tvt_data(X, y)
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    X_train_seq = create_sequences(X_train, timesteps=TIMESTEPS)
    X_val_seq = create_sequences(X_val, timesteps=TIMESTEPS)

    encoder, decoder, autoencoder = build_models(
        timesteps=TIMESTEPS, n_features=X_train_seq.shape[2]
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",  # surveille la loss validation
        factor=0.5,  # multiplie le learning rate par 0.5
        patience=N_PATIENCE,  # attend 5 epochs sans amélioration
        min_lr=1e-6,
        verbose=1,
    )

    history = autoencoder.fit(
        X_train_seq,
        X_train_seq,
        validation_data=(X_val_seq, X_val_seq),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            # early_stopping,
            reduce_lr
        ],
        shuffle=False,
    )

    autoencoder.save("artifacts/autoencoder.keras")
    encoder.save("artifacts/encoder.keras")
