import keras
import tensorflow as tf

class ReflectionPadding2D(keras.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, x) -> tf.Tensor:
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")

def compute_mean_std(x:tf.Tensor, epsilon=1e-5) -> tuple:
    axes = [1, 2]
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation

class AdaIN(keras.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.trainable = False
        self.build((None, None, 512))

    def call(self, content:tf.Tensor, style=tf.Tensor, alpha=1) -> tf.Tensor:
        content_mean, content_std = compute_mean_std(content)
        style_mean, style_std = compute_mean_std(style)
        t = style_std * ((content - content_mean) / content_std) + style_mean
        # Interpolate between content and style using alpha:
        # less alpha -> less style
        return alpha * t + ((1 - alpha) * content)

class Encoder(keras.Model):
    def __init__(self, input_shape=(None, None, 3)) -> None:
        super().__init__()
        self.input_shape = input_shape
        vgg19 = keras.applications.VGG19(
            include_top=False, 
            weights='imagenet', 
            input_shape=self.input_shape
        )
        vgg19.trainable = False
        self.layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        outputs = [vgg19.get_layer(name).output for name in self.layer_names]
        mini_vgg19 = keras.Model(vgg19.input, outputs)
        self.encoder = keras.Sequential()
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
                        bias_initializer=B_init
                    )
                )
                self.encoder.layers[-1].name = layer.name
            else:
                self.encoder.add(layer)
        self.encoder.trainable = False
        self.build(self.input_shape)

    def call(self, inputs, return_only_last=False) -> tf.Tensor:
        # relu4_1
        block4 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.layers[0].get_layer(self.layer_names[3]).output
        )
        # Last featuremap of encoder
        out4 = block4(inputs)
        # Inference speed optimization
        if return_only_last:
            return out4
        # relu1_1
        block1 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.layers[0].get_layer(self.layer_names[0]).output
        )
        # relu2_1
        block2 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.layers[0].get_layer(self.layer_names[1]).output
        )
        # relu3_1
        block3 = keras.Model(
            inputs=self.encoder.inputs,
            outputs=self.layers[0].get_layer(self.layer_names[2]).output
        )
        # Intermediate featuremaps
        out1 = block1(inputs)
        out2 = block2(inputs)
        out3 = block3(inputs)
        return out1, out2, out3, out4
    
class Decoder(keras.Model):
    def __init__(self, input_shape=(None, None, 512)) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.conv_config = dict(
            kernel_size=(3, 3),
            activation='relu',
            padding='valid'
        )
        # Mirrored encoder
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(self.input_shape),
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
        self.build(self.input_shape)

    def call(self, inputs) -> tf.Tensor:
        return self.decoder(inputs)
    
