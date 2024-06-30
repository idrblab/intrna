import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv2D, Concatenate,Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras.models import Sequential


def count_trainable_params(model):
    p = 0
    for layer in model.layers:
        if len(layer.trainable_variables) == 0:
            continue
        else:
            for variables in layer.trainable_variables:
                a = variables.shape.as_list()
                if len(a) == 1:
                    p += a[0]
                else:
                    p += a[0]*a[1]
    return p


def resnet_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


def Inception(inputs, units = 8, strides = 1):
    """
    naive google inception block
    """
    x1 = Conv2D(units, 5, padding='same', activation = 'relu', strides = strides)(inputs)
    x2 = Conv2D(units, 3, padding='same', activation = 'relu', strides = strides)(inputs)
    x3 = Conv2D(units, 1, padding='same', activation = 'relu', strides = strides)(inputs)
    outputs = Concatenate()([x1, x2, x3])    
    return outputs

def Inception2(inputs, units = 8, strides = 1):
    """
    naive google inception block
    """
    x1 = Conv2D(units, 5, padding='same', activation = 'relu', strides = strides)(inputs)
    x2 = Conv2D(units, 3, padding='same', activation = 'relu', strides = strides)(inputs)
    # x3 = Conv2D(units, 1, padding='same', activation = 'relu', strides = strides)(inputs)
    outputs = Concatenate()([x1, x2])    
    return outputs




def MolMapNet(input_shape,  
                n_outputs = 1, 
                conv1_kernel_size = 13,
                dense_layers = [128, 32], 
                dense_avf = 'relu', 
                last_avf = None):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    assert len(input_shape) == 3
    inputs = Input(input_shape)
    
    conv1 = Conv2D(48,  conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(inputs)
    
    conv1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(conv1) #p1
    
    incept1 = Inception(conv1, strides = 1, units = 32)
    
    incept1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept1) #p2
    
    incept2 = Inception(incept1, strides = 1, units = 64)
    
    #flatten
    x = GlobalMaxPool2D()(incept2)
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        
    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_auto_encoder(X_train, layers, batch_size=100, nb_epoch=100, activation='sigmoid', optimizer='adam'):
    trained_encoders = []
    trained_decoders = []
    X_train_tmp = np.copy(X_train)
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        ae = Sequential(
            [Dense(n_out, input_dim=X_train_tmp.shape[1], activation=activation, ),
             Dense(n_in, activation=activation),
             Dropout(0.5)]
        )
        ae.compile(loss='mean_squared_error', optimizer=optimizer)
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=nb_epoch, verbose=0, shuffle=True)
        # store trained encoder
        trained_encoders.append(ae.layers[0])
        trained_decoders.append(ae.layers[1])
        # update training data
        encoder = Sequential([ae.layers[0]])
        # encoder.evaluate(X_train_tmp, X_train_tmp, batch_size=batch_size)
        X_train_tmp = encoder.predict(X_train_tmp)

    return trained_encoders

def get_auto_encoders(X_train, batch_size=64):
    encoders_protein = train_auto_encoder(
        X_train=X_train[1],
        layers=[X_train[1].shape[1], 256, 128, 64], batch_size=batch_size)
    return encoders_protein


def dual_CNN_AE(encoders_protein, pro_coding_length, molmap1_size,
                    n_outputs = 1,
                    conv1_kernel_size = 13,
                    dense_layers = [128, 64], 
                    dense_avf = 'relu', 
                    last_avf = 'softmax',
                    dropout = 0.2):

    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept2)
    
    # NN for protein feature analysis
    xp_in_conjoint = Input(shape=(pro_coding_length,))
    xp_encoded = encoders_protein[0](xp_in_conjoint)
    xp_encoded = Dropout(dropout)(xp_encoded)
    xp_encoded = encoders_protein[1](xp_encoded)
    xp_encoded = Dropout(dropout)(xp_encoded)
    xp_encoder = encoders_protein[2](xp_encoded)
    xp_encoder = Dropout(dropout)(xp_encoder)
    xp_encoder = BatchNormalization()(xp_encoder)
    # xp_encoder = PReLU()(xp_encoder)
    xp_encoder = Dropout(dropout)(xp_encoder)

    # NN for RNA feature analysis
    # xr_in_conjoint = Input(shape=(rna_coding_length,))
    # xr_encoded = encoders_rna[0](xr_in_conjoint)
    # xr_encoded = Dropout(0.2)(xr_encoded)
    # xr_encoded = encoders_rna[1](xr_encoded)
    # xr_encoded = Dropout(0.2)(xr_encoded)
    # xr_encoded = encoders_rna[2](xr_encoded)
    # xr_encoder = Dropout(0.2)(xr_encoded)
    # xr_encoder = BatchNormalization()(xr_encoder)
    # # xr_encoder = PReLU()(xr_encoder)
    # xr_encoder = Dropout(0.2)(xr_encoder)

    ## concat
    x = Concatenate()([d_flat1, xp_encoder]) 
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, kernel_initializer='random_uniform', activation = dense_avf)(x)
        x = BatchNormalization()(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=[d_inputs1, xp_in_conjoint], outputs=outputs)

    # x_out_conjoint = concatenate([d_flat1, xp_encoder])
    # x_out_conjoint = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    # x_out_conjoint = BatchNormalization()(x_out_conjoint)
    # x_out_conjoint = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    # y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

    # model_conjoint = Model(inputs=[xp_in_conjoint, xr_in_conjoint], outputs=y_conjoint)


    return model

def singel_AE(encoders_protein, pro_coding_length,
                    n_outputs = 1,                    
                    dense_layers = [128, 64], 
                    dense_avf = 'relu', 
                    last_avf = 'softmax',
                    dropout = 0.2):

    # ## first inputs
    # d_inputs1 = Input(molmap1_size)
    # d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    # d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    # d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    # d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    # d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    # d_flat1 = GlobalMaxPool2D()(d_incept2)
    
    # NN for protein feature analysis
    xp_in_conjoint = Input(shape=(pro_coding_length,))
    xp_encoded = encoders_protein[0](xp_in_conjoint)
    xp_encoded = Dropout(dropout)(xp_encoded)
    xp_encoded = encoders_protein[1](xp_encoded)
    xp_encoded = Dropout(dropout)(xp_encoded)
    xp_encoder = encoders_protein[2](xp_encoded)
    xp_encoder = Dropout(dropout)(xp_encoder)
    xp_encoder = BatchNormalization()(xp_encoder)
    # xp_encoder = PReLU()(xp_encoder)
    x = Dropout(dropout)(xp_encoder)

    # NN for RNA feature analysis
    # xr_in_conjoint = Input(shape=(rna_coding_length,))
    # xr_encoded = encoders_rna[0](xr_in_conjoint)
    # xr_encoded = Dropout(0.2)(xr_encoded)
    # xr_encoded = encoders_rna[1](xr_encoded)
    # xr_encoded = Dropout(0.2)(xr_encoded)
    # xr_encoded = encoders_rna[2](xr_encoded)
    # xr_encoder = Dropout(0.2)(xr_encoded)
    # xr_encoder = BatchNormalization()(xr_encoder)
    # # xr_encoder = PReLU()(xr_encoder)
    # xr_encoder = Dropout(0.2)(xr_encoder)

    ## concat
    # x = Concatenate()([d_flat1, xp_encoder]) 
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, kernel_initializer='random_uniform', activation = dense_avf)(x)
        x = BatchNormalization()(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=xp_in_conjoint, outputs=outputs)

    # x_out_conjoint = concatenate([d_flat1, xp_encoder])
    # x_out_conjoint = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    # x_out_conjoint = BatchNormalization()(x_out_conjoint)
    # x_out_conjoint = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    # y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

    # model_conjoint = Model(inputs=[xp_in_conjoint, xr_in_conjoint], outputs=y_conjoint)


    return model

def singel_CNN(molmap1_size,
                    n_outputs = 1,
                    conv1_kernel_size = 13,
                    dense_layers = [128, 64], 
                    dense_avf = 'relu', 
                    last_avf = 'softmax',):

    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    x = GlobalMaxPool2D()(d_incept2)
    
    # # NN for protein feature analysis
    # xp_in_conjoint = Input(shape=(pro_coding_length,))
    # xp_encoded = encoders_protein[0](xp_in_conjoint)
    # xp_encoded = Dropout(dropout)(xp_encoded)
    # xp_encoded = encoders_protein[1](xp_encoded)
    # xp_encoded = Dropout(dropout)(xp_encoded)
    # xp_encoder = encoders_protein[2](xp_encoded)
    # xp_encoder = Dropout(dropout)(xp_encoder)
    # xp_encoder = BatchNormalization()(xp_encoder)
    # # xp_encoder = PReLU()(xp_encoder)
    # xp_encoder = Dropout(dropout)(xp_encoder)

    # NN for RNA feature analysis
    # xr_in_conjoint = Input(shape=(rna_coding_length,))
    # xr_encoded = encoders_rna[0](xr_in_conjoint)
    # xr_encoded = Dropout(0.2)(xr_encoded)
    # xr_encoded = encoders_rna[1](xr_encoded)
    # xr_encoded = Dropout(0.2)(xr_encoded)
    # xr_encoded = encoders_rna[2](xr_encoded)
    # xr_encoder = Dropout(0.2)(xr_encoded)
    # xr_encoder = BatchNormalization()(xr_encoder)
    # # xr_encoder = PReLU()(xr_encoder)
    # xr_encoder = Dropout(0.2)(xr_encoder)

    ## concat
    # x = Concatenate()([d_flat1, xp_encoder]) 
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, kernel_initializer='random_uniform', activation = dense_avf)(x)
        x = BatchNormalization()(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs = d_inputs1, outputs = outputs)

    # x_out_conjoint = concatenate([d_flat1, xp_encoder])
    # x_out_conjoint = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    # x_out_conjoint = BatchNormalization()(x_out_conjoint)
    # x_out_conjoint = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    # y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

    # model_conjoint = Model(inputs=[xp_in_conjoint, xr_in_conjoint], outputs=y_conjoint)


    return model

def MolMapDualPathNet(molmap1_size, 
                    molmap2_size, 
                    n_outputs = 1,
                    conv1_kernel_size = 13,
                    dense_layers = [256, 128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    """
    parameters
    ----------------------
    molmap1_size: w, h, c, shape of first molmap
    molmap2_size: w, h, c, shape of second molmap
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept2)

    
    ## second inputs
    f_inputs1 = Input(molmap2_size)
    f_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(f_inputs1)
    f_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_conv1) #p1
    f_incept1 = Inception(f_pool1, strides = 1, units = 32)
    f_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept1) #p2
    f_incept2 = Inception(f_pool2, strides = 1, units = 64)
    f_flat1 = GlobalMaxPool2D()(f_incept2)    
    
    ## concat
    x = Concatenate()([d_flat1, f_flat1]) 
    print('concat x shape: {}'.format(x.shape))
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=[d_inputs1, f_inputs1], outputs=outputs)
    
    return model

def MolMapDualPathNet_3inception(molmap1_size, 
                    molmap2_size, 
                    n_outputs = 1,
                    conv1_kernel_size = 13,
                    dense_layers = [256, 128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    """
    parameters
    ----------------------
    molmap1_size: w, h, c, shape of first molmap
    molmap2_size: w, h, c, shape of second molmap
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 16)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 32)
    d_pool3 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept2) #p2
    d_incept3 = Inception(d_pool3, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept3)

    
    ## second inputs
    f_inputs1 = Input(molmap2_size)
    f_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(f_inputs1)
    f_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_conv1) #p1
    f_incept1 = Inception(f_pool1, strides = 1, units = 16)
    f_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept1) #p2
    f_incept2 = Inception(f_pool2, strides = 1, units = 32)
    f_pool3 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept2) #p2
    f_incept3 = Inception(f_pool3, strides = 1, units = 64)
    f_flat1 = GlobalMaxPool2D()(f_incept3)    
    
    ## concat
    x = Concatenate()([d_flat1, f_flat1]) 
    print('concat x shape: {}'.format(x.shape))
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=[d_inputs1, f_inputs1], outputs=outputs)
    
    return model


def MolMapDualPathNet_dropout(molmap1_size, 
                    molmap2_size, 
                    n_outputs = 1,
                    conv1_kernel_size = 13,
                    dense_layers = [256, 128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    """
    parameters
    ----------------------
    molmap1_size: w, h, c, shape of first molmap
    molmap2_size: w, h, c, shape of second molmap
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept2)

    
    ## second inputs
    f_inputs1 = Input(molmap2_size)
    f_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(f_inputs1)
    f_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_conv1) #p1
    f_incept1 = Inception(f_pool1, strides = 1, units = 32)
    f_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept1) #p2
    f_incept2 = Inception(f_pool2, strides = 1, units = 64)
    f_flat1 = GlobalMaxPool2D()(f_incept2)    
    
    ## concat
    x = Concatenate()([d_flat1, f_flat1]) 
    print('concat x shape: {}'.format(x.shape))
    ## dense layer
    for index, units in enumerate(dense_layers):
        if index == 3:
            x = Dense(units, activation = dense_avf)(x)
        else:
            x = Dense(units, activation = dense_avf)(x)
            x = Dropout(0.2)(x)
    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=[d_inputs1, f_inputs1], outputs=outputs)
    
    return model


def MolMapOnePathNet(molmap1_size, 
                    n_outputs = 1,
                    conv1_kernel_size = 13,
                    dense_layers = [256, 128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    """
    parameters
    ----------------------
    molmap1_size: w, h, c, shape of first molmap
    molmap2_size: w, h, c, shape of second molmap
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept2)

    
    ## second inputs
    # f_inputs1 = Input(molmap2_size)
    # f_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(f_inputs1)
    # f_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_conv1) #p1
    # f_incept1 = Inception(f_pool1, strides = 1, units = 32)
    # f_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept1) #p2
    # f_incept2 = Inception(f_pool2, strides = 1, units = 64)
    # f_flat1 = GlobalMaxPool2D()(f_incept2)    
    
    ## concat
    # x = Concatenate()([d_flat1, f_flat1]) 
    x = d_flat1 
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    # model = tf.keras.Model(inputs=[d_inputs1, f_inputs1], outputs=outputs)
    model = tf.keras.Model(inputs=d_inputs1, outputs=outputs)
    
    return model

def MolMapAddPathNet(molmap_shape,  additional_shape,
                    n_outputs = 1,              
                    dense_layers = [128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    assert len(molmap_shape) == 3
    inputs = Input(molmap_shape)
    inputs_actvie_amount = Input(additional_shape)
    
    conv1 = Conv2D(48, 13, padding = 'same', activation='relu', strides = 1)(inputs)
    
    conv1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(conv1) #p1
    
    incept1 = Inception(conv1, strides = 1, units = 32)
    
    incept1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept1) #p2
    
    incept2 = Inception(incept1, strides = 1, units = 64)
    
    #flatten
    x = GlobalMaxPool2D()(incept2)
    x = tf.keras.layers.concatenate([x, inputs_actvie_amount], axis=-1)
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        
    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    
    model = tf.keras.Model(inputs=[inputs, inputs_actvie_amount], outputs=outputs)
    
    return model





def MolMapResNet(input_shape,
                 num_resnet_blocks = 8,
                n_outputs = 1, 
                dense_layers = [128, 32], 
                dense_avf = 'relu', 
                last_avf = None                
                ):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    num_resnet_blocks: int
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(input_shape) #input_shape = (24, 24, 3)
#     x = layers.Conv2D(32, 11, activation='relu')(inputs)
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(3)(x)
    x = Conv2D(64,  13, padding = 'same', activation='relu', strides = 1)(inputs)
    x = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x) #p1
    
    ## renet block 
    for i in range(num_resnet_blocks):
        x = resnet_block(x, 64, 3)

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.GlobalMaxPool2D()(x)

     ## dense layer
    for units in dense_layers:
        x = layers.Dense(units, activation = dense_avf)(x)
        
    
    
    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model



def MolMapDualResNet(molmap1_size,
                    molmap2_size,
                 num_resnet_blocks = 8,
                n_outputs = 1, 
                dense_layers = [128, 32], 
                dense_avf = 'relu', 
                last_avf = None                
                ):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    num_resnet_blocks: int
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    # inputs = tf.keras.Input(input_shape) #input_shape = (24, 24, 3)
#     x = layers.Conv2D(32, 11, activation='relu')(inputs)
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(3)(x)
    d_conv1 = Conv2D(64,  13, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_resnet1 = d_pool1
    ## renet block 
    for i in range(num_resnet_blocks):
        d_resnet1 = resnet_block(d_resnet1, 64, 3)

    d_conv2 = Conv2D(256, 3, activation='relu')(d_resnet1)
    d_pool2 = GlobalMaxPool2D()(d_conv2)

       
    ## second inputs
    g_inputs1 = Input(molmap2_size)
    # inputs = tf.keras.Input(input_shape) #input_shape = (24, 24, 3)
#     x = layers.Conv2D(32, 11, activation='relu')(inputs)
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(3)(x)
    g_conv1 = Conv2D(64,  13, padding = 'same', activation='relu', strides = 1)(g_inputs1)
    g_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(g_conv1) #p1
    g_resnet1 = g_pool1
    ## renet block 
    for i in range(num_resnet_blocks):
        g_resnet1 = resnet_block(g_resnet1, 64, 3)

    g_conv2 = Conv2D(256, 3, activation='relu')(g_resnet1)
    g_pool2 = GlobalMaxPool2D()(g_conv2)


    ## concat
    x = Concatenate()([d_pool2, g_pool2])
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=[d_inputs1, g_inputs1], outputs=outputs)


    return model

# ResNet 50 ==============================================================================
def identity_block(input_ten,kernel_size,filters):
    filters1,filters2,filters3 = filters
    
    x = Conv2D(filters1,(1,1))(input_ten)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2,kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters3,(1,1))(x)
    x = BatchNormalization()(x)
    
    x = layers.add([x,input_ten])
    x = Activation('relu')(x)
    return x

def conv_block(input_ten,kernel_size,filters,strides=(2,2)):
    filters1,filters2,filters3 = filters
    x = Conv2D(filters1,(1,1),strides=strides)(input_ten)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2,kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters3,(1,1))(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv2D(filters3,(1,1),strides=strides)(input_ten)
    shortcut = BatchNormalization()(shortcut)
    
    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(molmap1_size,
            molmap2_size,
                n_outputs = 1,
                dense_layers = [128, 32], 
                dense_avf = 'relu', 
                last_avf = None  
                ):

    tf.keras.backend.clear_session()
    ## first inputs
    d_input1 = Input(shape=molmap1_size)
    x = ZeroPadding2D((3,3))(d_input1) # (None, 34, 33, 14)

    x = Conv2D(64,(7,7),padding='same',strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),padding='same',strides=(1,1))(x)
    print('x.shape')
    print(x.shape)
    x =     conv_block(x,3,[64,64,256],strides=(1,1))
    x = identity_block(x,3,[64,64,256])
    x = identity_block(x,3,[64,64,256])
    
    x =     conv_block(x,3,[128,128,512])
    x = identity_block(x,3,[128,128,512])
    x = identity_block(x,3,[128,128,512])
    x = identity_block(x,3,[128,128,512])
    
    x =     conv_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    
    x =     conv_block(x,3,[512,512,2048])
    x = identity_block(x,3,[512,512,2048])
    x = identity_block(x,3,[512,512,2048])

    
    x = AveragePooling2D((5,5))(x)
    x = tf.keras.layers.Flatten()(x)

    ## second inputs
    g_input1 = Input(shape=molmap2_size)
    gx = ZeroPadding2D((3,3))(g_input1)
    
    gx = Conv2D(64,(7,7),padding='same',strides=(1,1))(gx)
    gx = BatchNormalization()(gx)
    gx = Activation('relu')(gx)
    gx = MaxPooling2D((3,3),padding='same',strides=(1,1))(gx)
    
    gx =     conv_block(gx,3,[64,64,256],strides=(1,1))
    gx = identity_block(gx,3,[64,64,256])
    gx = identity_block(gx,3,[64,64,256])
    
    gx =     conv_block(gx,3,[128,128,512])
    gx = identity_block(gx,3,[128,128,512])
    gx = identity_block(gx,3,[128,128,512])
    gx = identity_block(gx,3,[128,128,512])
    
    gx =     conv_block(gx,3,[256,256,1024])
    gx = identity_block(gx,3,[256,256,1024])
    gx = identity_block(gx,3,[256,256,1024])
    gx = identity_block(gx,3,[256,256,1024])
    gx = identity_block(gx,3,[256,256,1024])
    gx = identity_block(gx,3,[256,256,1024])
    
    gx =     conv_block(gx,3,[512,512,2048])
    gx = identity_block(gx,3,[512,512,2048])
    gx = identity_block(gx,3,[512,512,2048])
    
    gx = AveragePooling2D((5,5))(gx)
    gx = tf.keras.layers.Flatten()(gx)


    ## concat
    cx = Concatenate()([x, gx])
    
    ## dense layer
    for units in dense_layers:
        cx = Dense(units, activation = dense_avf)(cx)

    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(cx)
    model = tf.keras.Model(inputs=[d_input1, g_input1], outputs=outputs)
    
    return model




