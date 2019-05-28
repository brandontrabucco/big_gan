"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class GenResBlockUp(tf.keras.layers.Layer):
    
    def __init__(self, 
            image_size, 
            in_channels, 
            out_channels):
        super(GenResBlockUp, self).__init__()
        
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.linear_1 = tf.keras.layers.Dense(
            (image_size // 2) * (image_size // 2) * in_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size // 2, image_size // 2, in_channels])))
        self.conv_1 = tf.keras.layers.Conv2D(out_channels, 1, padding="same")
        
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.linear_2 = tf.keras.layers.Dense(
            (image_size // 2) * (image_size // 2) * out_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size // 2, image_size // 2, out_channels])))
        self.conv_2 = tf.keras.layers.Conv2D(out_channels, 3, padding="same")
        
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.linear_3 = tf.keras.layers.Dense(
            image_size * image_size * out_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size, image_size, out_channels])))
        self.conv_3 = tf.keras.layers.Conv2D(out_channels, 3, padding="same")
        
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.linear_4 = tf.keras.layers.Dense(
            image_size * image_size * out_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size, image_size, out_channels])))
        self.conv_4 = tf.keras.layers.Conv2D(out_channels, 1, padding="same")
        
        self.upsample_1 = tf.keras.layers.UpSampling2D(
            size=2, interpolation="bilinear")
        self.upsample_2 = tf.keras.layers.UpSampling2D(
            size=2, interpolation="bilinear")
        self.drop_channels = (lambda x: x[:, :, :, :out_channels])
            
    def call(self, x, z):
        h = tf.nn.relu(self.bn_1(self.linear_1(z) + x))
        h = tf.nn.relu(self.bn_2(self.linear_2(z) + self.conv_1(h)))
        
        h = self.upsample_1(h)
        x = self.upsample_2(self.drop_channels(x))
        
        h = tf.nn.relu(self.bn_3(self.linear_3(z) + self.conv_2(h)))
        h = tf.nn.relu(self.bn_4(self.linear_4(z) + self.conv_3(h)))
        return x + self.conv_4(h)
    
    
class GenResBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
            image_size, 
            num_channels):
        super(GenResBlock, self).__init__()
        
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.linear_1 = tf.keras.layers.Dense(
            image_size * image_size * num_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size, image_size, num_channels])))
        self.conv_1 = tf.keras.layers.Conv2D(num_channels, 1, padding="same")
        
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.linear_2 = tf.keras.layers.Dense(
            image_size * image_size * num_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size, image_size, num_channels])))
        self.conv_2 = tf.keras.layers.Conv2D(num_channels, 3, padding="same")
        
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.linear_3 = tf.keras.layers.Dense(
            image_size * image_size * num_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size, image_size, num_channels])))
        self.conv_3 = tf.keras.layers.Conv2D(num_channels, 3, padding="same")
        
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.linear_4 = tf.keras.layers.Dense(
            image_size * image_size * num_channels,
            activation=(lambda x: tf.reshape(x, [-1, 
                image_size, image_size, num_channels])))
        self.conv_4 = tf.keras.layers.Conv2D(num_channels, 1, padding="same")
            
    def call(self, x, z):
        h = tf.nn.relu(self.bn_1(self.linear_1(z) + x))
        h = tf.nn.relu(self.bn_2(self.linear_2(z) + self.conv_1(h)))
        
        h = tf.nn.relu(self.bn_3(self.linear_3(z) + self.conv_2(h)))
        h = tf.nn.relu(self.bn_4(self.linear_4(z) + self.conv_3(h)))
        return x + self.conv_4(h)
    
    
class DiscResBlockDown(tf.keras.layers.Layer):
    
    def __init__(self,  
            in_channels, 
            out_channels):
        super(DiscResBlockDown, self).__init__()
        
        self.conv_1 = tf.keras.layers.Conv2D(out_channels, 1, padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(out_channels, 3, padding="same")
        self.conv_3 = tf.keras.layers.Conv2D(out_channels, 3, padding="same")
        self.conv_4 = tf.keras.layers.Conv2D(out_channels, 1, padding="same")
        
        self.downsample_1 = tf.keras.layers.AveragePooling2D(
            pool_size=2, padding="same")
        self.downsample_2 = tf.keras.layers.AveragePooling2D(
            pool_size=2, padding="same")
    
        self.conv_add_channels = tf.keras.layers.Conv2D(
            out_channels - in_channels, 1, padding="same")
        self.add_channels = (lambda x: tf.concat([self.conv_add_channels(x), 
            x], 3))
            
    def call(self, x):
        h = tf.nn.relu(x)
        h = tf.nn.relu(self.conv_1(h))
        
        h = tf.nn.relu(self.conv_2(h))
        h = tf.nn.relu(self.conv_3(h))
        
        h = self.downsample_1(h)
        x = self.add_channels(self.downsample_2(x))
        return x + self.conv_4(h)
    
    
class DiscResBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
            num_channels):
        super(DiscResBlock, self).__init__()
        
        self.conv_1 = tf.keras.layers.Conv2D(num_channels, 1, padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(num_channels, 3, padding="same")
        self.conv_3 = tf.keras.layers.Conv2D(num_channels, 3, padding="same")
        self.conv_4 = tf.keras.layers.Conv2D(num_channels, 1, padding="same")
            
    def call(self, x):
        h = tf.nn.relu(x)
        h = tf.nn.relu(self.conv_1(h))
        
        h = tf.nn.relu(self.conv_2(h))
        h = tf.nn.relu(self.conv_3(h))
        return x + self.conv_4(h)
    
    
class ScaledDotProductAttention(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, hidden_size, output_size):
        super(ScaledDotProductAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.query_map = tf.keras.layers.Dense(hidden_size * num_heads, 
            use_bias=False)
        self.key_map = tf.keras.layers.Dense(hidden_size * num_heads, 
            use_bias=False)
        self.value_map = tf.keras.layers.Dense(hidden_size * num_heads, 
            use_bias=False)
        self.output_map = tf.keras.layers.Dense(output_size, 
            use_bias=False)
        
    def call(self, queries, keys, values):
        batch_size, num_queries, sequence_length = (tf.shape(queries)[0], 
            tf.shape(queries)[1], tf.shape(values)[1])
        Q, K, V = (self.query_map(queries), self.key_map(keys), 
            self.value_map(values))
        
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, 
            self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, 
            self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, 
            self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        
        Z = tf.sqrt(float(self.hidden_size))
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(
            K, [0, 1, 3, 2])) / Z), V)
        return self.output_map(tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [
            batch_size, num_queries, self.num_heads * self.hidden_size]))
    
    
class Generator(tf.keras.models.Model):
    
    def __init__(self, num_classes, embedding_size, multiplier):
        super(Generator, self).__init__()
        
        self.embeddings_map = tf.keras.layers.Embedding(
            num_classes, embedding_size)
        self.linear = tf.keras.layers.Dense(
            1 * 1 * 16 * multiplier,
            activation=(lambda x: tf.reshape(x, [-1, 
                1, 1, 16 * multiplier])))
        
        self.resblock_1 = GenResBlock(1, 16 * multiplier)
        self.resblock_up_1 = GenResBlockUp(2, 16 * multiplier, 
            16 * multiplier)
        
        self.resblock_2 = GenResBlock(2, 16 * multiplier)
        self.resblock_up_2 = GenResBlockUp(4, 16 * multiplier, 
            8 * multiplier)
        
        self.resblock_3 = GenResBlock(4, 8 * multiplier)
        self.resblock_up_3 = GenResBlockUp(8, 8 * multiplier, 
            4 * multiplier)
        
        self.resblock_4 = GenResBlock(8, 4 * multiplier)
        self.resblock_up_4 = GenResBlockUp(16, 4 * multiplier, 
            2 * multiplier)
        
        self.flatten = (lambda x: tf.reshape(x, [-1, 
            16 * 16, 2 * multiplier]))
        self.non_local = ScaledDotProductAttention(2, multiplier, 
            2 * multiplier)
        self.unflatten = (lambda x: tf.reshape(x, [-1, 
            16, 16, 2 * multiplier]))
        
        self.resblock_5 = GenResBlock(16, 2 * multiplier)
        self.resblock_up_5 = GenResBlockUp(32, 2 * multiplier, 
            multiplier)
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(3, 3, padding="same")
        
    def call(self, c):
        embedded_c = self.embeddings_map(c)
        z = tf.concat([
            tf.random_normal([tf.shape(embedded_c)[0], 128]),
            embedded_c], 1)
        
        h = self.resblock_1(self.linear(z), z)
        h = self.resblock_up_1(h, z)
        
        h = self.resblock_2(h, z)
        h = self.resblock_up_2(h, z)
        
        h = self.resblock_3(h, z)
        h = self.resblock_up_3(h, z)
        
        h = self.resblock_4(h, z)
        h = self.resblock_up_4(h, z)
        
        h = self.flatten(h)
        h = h + self.non_local(h, h, h)
        h = self.unflatten(h)
        
        h = self.resblock_5(h, z)
        h = self.resblock_up_5(h, z)
        return tf.nn.tanh(self.conv(tf.nn.relu(self.bn(h))))
    
    
class Discriminator(tf.keras.models.Model):
    
    def __init__(self, num_classes, multiplier):
        super(Discriminator, self).__init__()
        self.conv = tf.keras.layers.Conv2D(multiplier, 3, padding="same")
        self.dense = tf.keras.layers.Dense(num_classes + 1)
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        
        self.resblock_down_1 = DiscResBlockDown(multiplier, 
            2 * multiplier)
        self.resblock_1 = DiscResBlock(2 * multiplier)
        
        self.flatten = (lambda x: tf.reshape(x, [-1, 
            16 * 16, 2 * multiplier]))
        self.non_local = ScaledDotProductAttention(2, multiplier, 
            2 * multiplier)
        self.unflatten = (lambda x: tf.reshape(x, [-1, 
            16, 16, 2 * multiplier]))
        
        self.resblock_down_2 = DiscResBlockDown(2 * multiplier, 
            4 * multiplier)
        self.resblock_2 = DiscResBlock(4 * multiplier)
        
        self.resblock_down_3 = DiscResBlockDown(4 * multiplier, 
            8 * multiplier)
        self.resblock_3 = DiscResBlock(8 * multiplier)
        
        self.resblock_down_4 = DiscResBlockDown(8 * multiplier, 
            16 * multiplier)
        self.resblock_4 = DiscResBlock(16 * multiplier)
        
        self.resblock_down_5 = DiscResBlockDown(16 * multiplier, 
            16 * multiplier)
        self.resblock_5 = DiscResBlock(16 * multiplier)
        
    def call(self, x):
        h = self.conv(x)
        h = self.resblock_down_1(h)
        h = self.resblock_1(h)
        
        h = self.flatten(h)
        h = h + self.non_local(h, h, h)
        h = self.unflatten(h)
        
        h = self.resblock_down_2(h)
        h = self.resblock_2(h)
        
        h = self.resblock_down_3(h)
        h = self.resblock_3(h)
        
        h = self.resblock_down_4(h)
        h = self.resblock_4(h)
        
        h = self.resblock_down_5(h)
        h = self.resblock_5(h)
        return self.dense(self.global_pooling(tf.nn.relu(h)))
    
    
if __name__ == "__main__":
    

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    id_to_name = [
        "airplane", 
        "automobile", 
        "bird", 
        "cat", 
        "deer", 
        "dog", 
        "frog", 
        "horse", 
        "ship", 
        "truck"]


    dataset_train = tf.data.Dataset.from_tensor_slices({
        "images": (x_train * 2.0)- 1.0, 
        "labels": y_train[:, 0]}).apply(
            tf.data.experimental.shuffle_and_repeat(1000)).batch(32).apply(
                tf.data.experimental.prefetch_to_device("/gpu:0", buffer_size=2))

    dataset_test = tf.data.Dataset.from_tensor_slices({
        "images": (x_test * 2.0) - 1.0, 
        "labels": y_test[:, 0]}).apply(
            tf.data.experimental.shuffle_and_repeat(1000)).batch(32).apply(
                tf.data.experimental.prefetch_to_device("/gpu:0", buffer_size=2))
    
    
    #generator = Generator(10, 128, 32)
    discriminator = Discriminator(10, 32)
    
    
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
    
    
    global_step = tf.train.get_or_create_global_step()
    iterator = dataset_train.make_initializable_iterator()
    sess.run(iterator.initializer)
    sess.run(tf.variables_initializer([global_step]))

    train_dict = iterator.get_next()
    real_x = tf.cast(train_dict["images"], tf.float32)
    y = tf.cast(train_dict["labels"], tf.int32)


    #fake_x = generator(y)
    real_logits = discriminator(real_x)
    #fake_logits = discriminator(fake_x)

    D_loss_one = tf.losses.sparse_softmax_cross_entropy(
        y, real_logits)
    #D_loss_two = tf.losses.sparse_softmax_cross_entropy(
    #    tf.fill([tf.shape(y)[0]], 10), fake_logits)

    D_loss = D_loss_one# + D_loss_two
    #G_loss = tf.losses.sparse_softmax_cross_entropy(
    #    y, fake_logits)


    D_optimizer = tf.train.GradientDescentOptimizer(0.0002)
    D_gradients = D_optimizer.compute_gradients(D_loss, 
        var_list=discriminator.trainable_variables)
    D_capped_gradients = [(tf.clip_by_norm(gradient, 1.0), variable) 
        for gradient, variable in D_gradients]
    D_step = D_optimizer.apply_gradients(D_capped_gradients)

    #G_optimizer = tf.train.GradientDescentOptimizer(0.00005)
    #G_gradients = G_optimizer.compute_gradients(G_loss, 
    #    var_list=generator.trainable_variables)
    #G_capped_gradients = [(tf.clip_by_norm(gradient, 1.0), variable) 
    #    for gradient, variable in G_gradients]
    #G_step = G_optimizer.apply_gradients(G_capped_gradients, 
    #    global_step=global_step)


    update_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    sess.run(tf.variables_initializer(
        [global_step]))
    sess.run(tf.variables_initializer(
        D_optimizer.variables()))
    #sess.run(tf.variables_initializer(
    #    G_optimizer.variables()))
    sess.run(tf.variables_initializer(
        discriminator.variables))
    #sess.run(tf.variables_initializer(
    #    generator.variables))

    
    D_saver = tf.train.Saver(var_list=discriminator.variables)
    #G_saver = tf.train.Saver(var_list=generator.variables)


    for i in range(1000):
    
        dloss, _d = sess.run([D_loss, D_step])
        #gloss, _g, _u = sess.run([G_loss, G_step, update_step])
        iteration = sess.run(global_step)

        print("Iteration {:05d} loss was {:02.5f}".format(
            iteration, dloss + gloss))


    D_saver.save(sess, "discriminator.ckpt", global_step=global_step)
    G_saver.save(sess, "generator.ckpt", global_step=global_step)
