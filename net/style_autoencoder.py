import keras
import tensorflow as tf
import numpy as np
from skimage.exposure import match_histograms
from net.net_modules import Decoder, Encoder, AdaIN, compute_mean_std

class StyleTransfer(keras.Model):
    def __init__(self, style_weight=10, content_weight=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.adain = AdaIN()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.resizer = keras.layers.Resizing
        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.lr_tracker = keras.metrics.Mean(name='learning_rate')  
        self.imgnet_mean_bgr = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)

    def get_build_config(self) -> dict:
        build_config = super().get_build_config()
        return build_config

    def build_from_config(self, config:dict) -> None:
        self.build(config["input_shape"])
        
    def compile(self, optimizer, loss_fn, **kwargs) -> None:
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def get_compile_config(self) -> dict:
        return {
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
        }
        
    def compile_from_config(self, config:dict) -> None:
        optimizer = keras.utils.deserialize_keras_object(config["optimizer"])
        loss_fn = keras.utils.deserialize_keras_object(config["loss_fn"])
        self.compile(optimizer=optimizer, loss_fn=loss_fn)

    def preprocess_input(self, img:tf.Tensor, resize=None) -> tuple:
        # Decentered RGB (uint8) -> centered BGR (float32)
        img = tf.reverse(img, axis=[-1])
        img = tf.cast(img, tf.float32) - self.imgnet_mean_bgr
        # Resize for inference
        if resize:
            img = self.resizer(*resize)(img)
        return img

    def postprocess_output(self, reconstructed_image:tf.Tensor, output_shape:tuple) -> tf.Tensor:
        # Centered (float32) -> decenter -> clip to [0, 255] -> uint8
        decentered_img = tf.cast(tf.clip_by_value((reconstructed_image + self.imgnet_mean_bgr), 0, 255), tf.uint8)
        # Resize to original shape of content image
        resized_img = self.resizer(*output_shape)(decentered_img)
        # BGR -> RGB
        return tf.reverse(resized_img, axis=[-1])

    def train_step(self, inputs:tf.Tensor) -> dict:
        loss_content = 0.0
        loss_style = 0.0
        # Compute losses
        with tf.GradientTape() as tape:
            style_encoded, content_encoded, t, reconstructed_image = self(inputs)
            reconstructed_img_features = self.encoder(reconstructed_image)
            # Compute content loss between 'block4_conv1' and encoder(adain(c, s))
            loss_content = self.loss_fn(reconstructed_img_features[-1], t)
            # Compute style loss
            for inp, out in zip(style_encoded, reconstructed_img_features):
                mean_inp, std_inp = compute_mean_std(inp)
                mean_out, std_out = compute_mean_std(out)
                loss_style += self.loss_fn(mean_out, mean_inp) + self.loss_fn(std_out, std_inp)
            # Apply loss weights, concat losses into one loss func
            loss_style *= self.style_weight
            loss_content *= self.content_weight
            total_loss = loss_style + loss_content

        # Compute gradients and optimize the decoder
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        self.lr_tracker.update_state(self.optimizer.learning_rate)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "learning_rate": self.lr_tracker.result()
        }
    # Train
    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        content, style = inputs[0], inputs[1]
        # RGB -> BGR -> center by mean
        content = self.preprocess_input(content)
        style = self.preprocess_input(style)
        content_encoded = self.encoder(content)
        style_encoded = self.encoder(style)
        # Compute the AdaIN target feature maps. [-1] is 'block4_conv1' ('relu4_1' in original paper)
        t = self.adain((content_encoded[-1], style_encoded[-1]))
        # Generate the neural style transferred image
        reconstructed_image = self.decoder(t)
        return style_encoded, content_encoded, t, reconstructed_image
    # Inference
    def predict(self, content:tf.Tensor, style:tf.Tensor, alpha=1, save_content_colors=False) -> tf.Tensor:
        content_shape = content.shape[1:3]
        style_shape = style.shape[1:3]
        if min(content_shape) > 1079:
            output_shape = list(map(lambda x: int(x / 1.5), content_shape))
        else:
            output_shape = content_shape
        resize = (720, 720)
        content = self.preprocess_input(content, resize=resize)
        style = self.preprocess_input(style, resize=resize)
        if save_content_colors:
            # match_histograms method takes only numpy arrays, not tensors
            if isinstance(content, tf.Tensor) and isinstance(style, tf.Tensor):
                content, style = content.numpy(), style.numpy()
            else:
                content, style = np.asarray(content), np.asarray(style)
            style = match_histograms(style, content, channel_axis=-1)
        # Inference func returns only one output from 'block4_conv1' / 'relu4_1'
        content_encoded = self.encoder(content, inference=True)
        style_encoded = self.encoder(style, inference=True)
        # AdaIN feature map
        t = self.adain((content_encoded, style_encoded))
        # Interpolate between content and style using alpha
        interpolated_features = (1 - alpha)*content_encoded + alpha*t
        reconstructed_image = self.decoder(interpolated_features)
        output = self.postprocess_output(reconstructed_image, output_shape=output_shape)
        return output
    
    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
            self.lr_tracker
        ]
        