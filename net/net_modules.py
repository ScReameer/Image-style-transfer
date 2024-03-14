import keras
import tensorflow as tf

class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        return tf.pad(inputs, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")

def compute_mean_std(inputs:tf.Tensor, epsilon=1e-5) -> tuple:
    axes = [1, 2]
    mean, variance = tf.nn.moments(inputs, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation

class AdaIN(keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.trainable = False
        self.build((None, None, 512))
        
    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        content, style = inputs[0], inputs[1]
        content_mean, content_std = compute_mean_std(content)
        style_mean, style_std = compute_mean_std(style)
        t = style_std * ((content - content_mean) / content_std) + style_mean
        return t

class Encoder(keras.layers.Layer):
    def __init__(self, input_shape=(None, None, 3)) -> None:
        super().__init__()
        self.trainable = False
        vgg19 = keras.applications.VGG19(
            include_top=False, 
            weights='imagenet', 
            input_shape=input_shape
        )
        vgg19.trainable = False
        self.layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        outputs = [vgg19.get_layer(name).output for name in self.layer_names]
        mini_vgg19 = keras.Model(vgg19.input, outputs)
        self.encoder = keras.Sequential(name='encoder')
        for layer in mini_vgg19.layers:
            if isinstance(layer, keras.layers.Conv2D):
                # Weights of vgg19 conv layer
                W_init = keras.initializers.Constant(layer.get_weights()[0])
                # Bias of vgg19 conv layer
                B_init = keras.initializers.Constant(layer.get_weights()[1])
                self.encoder.add(ReflectionPadding2D())
                self.encoder.add(
                    keras.layers.Conv2D(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding='valid',
                        dilation_rate=layer.dilation_rate,
                        activation=layer.activation,
                        kernel_initializer=W_init,
                        bias_initializer=B_init,
                        name=layer.name
                    )
                )
            else:
                self.encoder.add(layer)
        self.encoder.trainable = False
        self.block1 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.encoder.get_layer(self.layer_names[0]).output
        )
        self.block2 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.encoder.get_layer(self.layer_names[1]).output
        )
        self.block3 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.encoder.get_layer(self.layer_names[2]).output
        )
        self.block4 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.encoder.get_layer(self.layer_names[3]).output
        )
        self.build(input_shape)

    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        out1 = self.block1(inputs)
        out2 = self.block2(inputs)
        out3 = self.block3(inputs)
        out4 = self.block4(inputs)
        return out1, out2, out3, out4
    
    def inference(self, inputs:tf.Tensor):
        out4 = self.block4(inputs)
        return out4
    
class Decoder(keras.layers.Layer):
    def __init__(self, input_shape=(None, None, 512)) -> None:
        super().__init__()
        self.conv_config = dict(
            kernel_size=(3, 3),
            activation='relu',
            padding='valid'
        )
        # Mirrored encoder
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape),
            ReflectionPadding2D(),
            keras.layers.Conv2D(256, **self.conv_config),
            keras.layers.UpSampling2D(),
            ReflectionPadding2D(),
            keras.layers.Conv2D(256, **self.conv_config),
            ReflectionPadding2D(),
            keras.layers.Conv2D(256, **self.conv_config),
            ReflectionPadding2D(),
            keras.layers.Conv2D(256, **self.conv_config),
            ReflectionPadding2D(),
            keras.layers.Conv2D(128, **self.conv_config),
            keras.layers.UpSampling2D(),
            ReflectionPadding2D(),
            keras.layers.Conv2D(128, **self.conv_config),
            ReflectionPadding2D(),
            keras.layers.Conv2D(64, **self.conv_config),
            keras.layers.UpSampling2D(),
            ReflectionPadding2D(),
            keras.layers.Conv2D(64, **self.conv_config),
            ReflectionPadding2D(),
            keras.layers.Conv2D(3, kernel_size=(3, 3), padding='valid')
        ], name='decoder')
        self.build(input_shape)

    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        return self.decoder(inputs)
    
