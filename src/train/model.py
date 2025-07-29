import tensorflow as tf
from tensorflow.keras import layers

class LayerNormalization(layers.Layer):
    
    def __init__(self, n_features, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.n_features = n_features
        self.epsilon = epsilon
        # 1 x n_features
        self.gamma = self.add_weight(name='gamma', shape=(n_features,), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(n_features,), initializer='zeros', trainable=True)
        
    def call(self, x):
        # batch_size x n_features --> 1 x n_features 
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        # batch_size x n_features --> 1 x n_features
        std = tf.math.reduce_std(x, axis=0, keepdims=True)
        # batch_size x n_features --> batch_size x n_features
        normalized_x = (x - mean) / (std + self.epsilon)
        return self.gamma * normalized_x + self.beta

class DNN(tf.keras.Model):

    def __init__(self, 
                 input_shape: int = 33, 
                 n_hidden: int = 5, 
                 output_shape: int = 2, 
                 output_bias: tf.Tensor = None,
                 n_hidden_neurons: list = [33, 33, 33, 33, 10],
                 dropout: float = 0.0):
        
        super(DNN, self).__init__()
        self.hidden_layers = []
        self.layer_norms = []
        for i in range(n_hidden):
            if i == 0:
                self.hidden_layers.append(layers.Dense(n_hidden_neurons[i], input_shape=(input_shape,), name=f'hidden_layer_{i:02d}', kernel_initializer='glorot_uniform'))
            else:
                self.hidden_layers.append(layers.Dense(n_hidden_neurons[i], name=f'hidden_layer_{i:02d}', kernel_initializer='glorot_uniform'))
            self.layer_norms.append(LayerNormalization(n_hidden_neurons[i]))
        self.dropout = layers.Dropout(dropout)
        self.gelu = layers.Activation('gelu')
        self.softmax = layers.Softmax(axis=-1)
        output_bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else 'zeros'
        self.output_layer = layers.Dense(output_shape, bias_initializer=output_bias_initializer, name='output_layer')
        
    def call(self, x, training=False):
        """
        x: Input features, batch_size x 33
        """
        
        for i, layer in enumerate(self.hidden_layers):
            # batch_size x 33 --> batch_size x n_hidden_neurons[i]
            x = layer(x)
            # batch_size x n_hidden_neurons[i] --> batch_size x n_hidden_neurons[i]
            x = self.layer_norms[i](x)
            # batch_size x n_hidden_neurons[i] --> batch_size x n_hidden_neurons[i]
            x = self.gelu(x)
            x = self.dropout(x, training=training)
        
        # batch_size x 10 --> batch_size x 2
        x = self.output_layer(x)
        return self.softmax(x)
