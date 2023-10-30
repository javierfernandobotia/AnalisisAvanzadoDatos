def autoencoder(layers, function_activation = 'relu', init = 'glorot_uniform'):
    n_stacks = len(layers) - 1    
    input_data = Input(shape=(layers[0],), name='input')
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks-1):
        x = Dense(layers[i + 1], activation = function_activation,  kernel_initializer=init, name='encoder_%d' % i)(x)
    
    # latent hidden layer
    encoded = Dense(layers[-1], kernel_initializer=init, name ='encoder_%d' % (n_stacks - 1))(x)
    x = encoded
    
    # internal layers of decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(layers[i], activation = function_activation, kernel_initializer=init, name = 'decoder_%d' % i)(x)
    
    # decoder output
    x = Dense(layers[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    autoencoder_model = Model(inputs=input_data, outputs=decoded, name = 'autoencoder')
    encoder_model = Model(inputs=input_data, outputs=encoded, name = 'encoder')
    
    return autoencoder_model, encoder_model