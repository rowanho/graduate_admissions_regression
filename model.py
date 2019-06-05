def build_model(dataset,hidden_size,learning_rate):
    model = keras.Sequential([
        layers.Dense(hidden_size,activation=tf.nn.relu, input_shape = [len(dataset.keys())]),
        layers.Dense(hidden_size, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])
    return model
