import tensorflow as tf
import numpy as np
import cupy as cp
from sklearn import metrics
import os, csv, cv2, random
from PIL import Image

class Datamanage:    
    def image_manage(self, img_file, flag):
        if flag == 'train':
            img = Image.open('train/' + img_file)
            img_size = img.resize((40, 40), Image.ANTIALIAS)
            img_arr = np.array(img_size)
            a = random.randint(0, 8)
            b = random.randint(0, 8)
            cropped = img_arr[a:a+32, b:b+32]
            f = random.randint(0, 1)
            if f == 1:
                cropped = cv2.flip(cropped, 1)
            img_result = cp.reshape(cropped, (1, -1))
        else:
            img = Image.open('train/' + img_file) # 这里的路径需要注意，训练和测试的时候是不一样的
            img_size = img.resize((40, 40), Image.ANTIALIAS)
            img_arr = np.array(img_size)
            cropped = img_arr[4:36, 4:36]
            img_result = cp.reshape(cropped, (1, -1))
        return img_result

    def read_and_convert(self, filelist, flag):
        if flag == 'train':
            data = self.image_manage(filelist[0], 'train')
            for i in range(1, len(filelist)):
                img = filelist[i] 
                data =np.concatenate((data, self.image_manage(img, 'train')), axis=0)
        else:
            data = self.image_manage(filelist[0], 'test')
            for i in range(1, len(filelist)):
                img = filelist[i] 
                data =np.concatenate((data, self.image_manage(img, 'test')), axis=0)
        return data

    def label_manage(self, csv_path, num_classes):
        label = self.csv_read(csv_path)
        total_y = np.zeros((len(label), num_classes))
        for i in range(len(label)):
            if label[i]=='airplane': total_y[i][0] = 1
            elif label[i]=='automobile': total_y[i][1] = 1
            elif label[i]=='bird': total_y[i][2] = 1
            elif label[i]=='cat': total_y[i][3] = 1
            elif label[i]=='deer': total_y[i][4] = 1
            elif label[i]=='dog': total_y[i][5] = 1
            elif label[i]=='frog': total_y[i][6] = 1
            elif label[i]=='horse': total_y[i][7] = 1
            elif label[i]=='ship': total_y[i][8] = 1
            elif label[i]=='truck': total_y[i][9] = 1
        return total_y

    def csv_read(self, data_path):
        label = []
        with open(data_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                label.append(row[1])
            new_label = np.reshape(label[1:], (-1, 1))
        return new_label

    def csv_write(self, data):
        f = open('result.csv', 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "label"])
        for i in range(len(data)):
            csv_writer.writerow([str(i+1), data[i]])

# 切换版本请自行取消增加注释
class Resnet():
    '''
    Resnet18 与 Resnet32 -----v1
    '''
    # def residual(self, inputs, num_channels, training, use_1x1conv=False, strides=1):
    #     outputs = tf.layers.conv2d(inputs=inputs, filters=num_channels, kernel_size=3, padding='same', 
    #                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    #                             strides=strides, activation=None, use_bias=False) 
    #     outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
    #     outputs = tf.nn.relu(outputs)
    
    #     outputs = tf.layers.conv2d(inputs=outputs, filters=num_channels, kernel_size=3, padding='same', 
    #                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), 
    #                             strides=1, activation=None, use_bias=False)
    #     outputs = tf.layers.batch_normalization(inputs=outputs, training=training)

    #     if use_1x1conv:
    #         inputs = tf.layers.conv2d(inputs=inputs, filters=num_channels, kernel_size=1, padding='same', 
    #                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
    #                                 strides=strides, activation=None, use_bias=False)
    #         inputs = tf.layers.batch_normalization(inputs=inputs, training=training) 
        
    #     result = tf.add(inputs, outputs)
    #     return tf.nn.relu(result)
    '''
    Resnet18 与 Resnet32 -----v2
    '''
    def residual(self, inputs, num_channels, training, use_1x1conv=False, strides=1):
        outputs = tf.layers.batch_normalization(inputs=inputs, training=training)
        outputs = tf.nn.relu(outputs)

        outputs = tf.layers.conv2d(inputs=outputs, filters=num_channels, kernel_size=3, padding='same', 
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                strides=strides, activation=None, use_bias=False) 
        outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
        outputs = tf.nn.relu(outputs)
        
        outputs = tf.layers.conv2d(inputs=outputs, filters=num_channels, kernel_size=3, padding='same', 
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), 
                                strides=1, activation=None, use_bias=False)

        if use_1x1conv:
            inputs = tf.layers.conv2d(inputs=inputs, filters=num_channels, kernel_size=1, padding='same', 
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                    strides=strides, activation=None, use_bias=False) 
            inputs = tf.layers.batch_normalization(inputs=inputs, training=training)

        return tf.add(inputs, outputs)
    '''
    Resnet50 与 Resnet101 与 Resnet152
    '''
    # def residual(self, inputs, num_channels, training, use_1x1conv=False, strides=1):
    #     outputs = tf.layers.conv2d(inputs=inputs, filters=num_channels, kernel_size=1, padding='same', 
    #                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    #                             strides=strides, activation=None, use_bias=False) 
    #     outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
    #     outputs = tf.nn.relu(outputs)

    #     outputs = tf.layers.conv2d(inputs=outputs, filters=num_channels, kernel_size=3, padding='same', 
    #                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    #                             strides=1, activation=None, use_bias=False)    
    #     outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
    #     outputs = tf.nn.relu(outputs)
    
    #     outputs = tf.layers.conv2d(inputs=outputs, filters=num_channels*4, kernel_size=1, padding='same', 
    #                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    #                             strides=1, activation=None, use_bias=False)   
    #     outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
        
    #     if use_1x1conv or inputs.shape!=outputs.shape:
    #         inputs = tf.layers.conv2d(inputs=inputs, filters=num_channels*4, kernel_size=1, padding='same', 
    #                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
    #                                 strides=strides, activation=None, use_bias=False)
    #         inputs = tf.layers.batch_normalization(inputs=inputs, training=training)
        
    #     result = tf.add(inputs, outputs)
    #     return tf.nn.relu(result)
    
    def block(self, inputs, num_channels, num_residuals, training, first_block=False):
        outputs = inputs
        for i in range(num_residuals):
            if i == 0 and not first_block:
                outputs = self.residual(outputs, num_channels, training=training, use_1x1conv=True, strides=2)
            else:
                outputs = self.residual(outputs, num_channels, training=training)
        return outputs

def train():
    '''
    参数设置
    '''
    num_classes = 10  # 输出大小
    input_size = 32*32*3  # 输入大小
    training_iterations = 30000 # 训练轮数
    weight_decay = 2e-4 # 权重衰减系数
    ver = 2 # 版本号 1 or 2
    manage = Datamanage()
    resnet = Resnet()
    '''
    数据读取
    '''
    path = 'train/'       
    data = os.listdir(path)
    data.sort(key=lambda x:int(x.split('.')[0]))
    label = manage.label_manage('train.csv', num_classes)
    x_train = data[:49000]; x_test = data[49000:]
    y_train = label[:49000]; y_test = label[49000:] 
    y_test = [np.argmax(x) for x in y_test]
    '''
    网络搭建
    '''
    X = tf.placeholder(tf.float32, shape = [None, input_size], name='x')
    Y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
    training = tf.placeholder(tf.bool, name="training")

    input_images = tf.reshape(X, [-1, 32, 32, 3])
    
    input_images = tf.image.per_image_standardization(input_images) # 图片标准化处理
    print(input_images.shape)
    inputs = tf.layers.conv2d(inputs=input_images, filters=64, kernel_size=3, strides=1, padding='same', 
                            activation=None, use_bias=False)
 
    if ver == 1:
        inputs = tf.nn.relu(tf.layers.batch_normalization(inputs, training=training)) 
    
    max_pool = tf.layers.max_pooling2d(inputs, pool_size=3, strides=2, padding='same')	    

    '''
    resnet18 [2, 2, 2, 2]
    resnet32 [3, 4, 6, 3]
    resnet50 [3, 4, 6, 3]
    resnet101 [3, 4, 23, 3]
    resnet152 [3, 8, 36, 3] 
    '''
    num_residuals = [2, 2, 2, 2]
    blk = resnet.block(max_pool, 64, num_residuals[0], training=training, first_block=True)
    blk = resnet.block(blk, 128, num_residuals[1], training=training)
    blk = resnet.block(blk, 256, num_residuals[2], training=training)
    blk = resnet.block(blk, 512, num_residuals[3], training=training)

    if ver == 2:
        inputs = tf.nn.relu(tf.layers.batch_normalization(inputs, training=training))

    pool = tf.layers.average_pooling2d(blk, pool_size=2, strides=2, padding='same')
    
    final_opt = tf.layers.dense(inputs=pool, units=10)
    tf.add_to_collection('pred_network', final_opt)

    # 学习率衰减
    global_step = tf.Variable(0, trainable=False)
    '''
    分段学习率
    '''
    boundaries = [10000, 15000, 20000, 25000]
    values = [0.1, 0.05, 0.01, 0.005, 0.001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    '''
    持续衰减
    '''
    # initial_learning_rate = 0.002 # 初始学习率
    # learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate, global_step=global_step, decay_steps=200, decay_rate=0.95)

    # 对输出层计算交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = loss + l2_loss

    # 定义优化器
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.minimize(loss, global_step=global_step)

    # 初始化
    sess = tf.Session() 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    '''
    训练
    '''
    for i in range(training_iterations):
        start_step = i*128 % 49000
        stop_step = start_step + 128
        
        batch_x, batch_y = x_train[start_step:stop_step], y_train[start_step:stop_step]
        batch_x = manage.read_and_convert(batch_x, 'train')
        
        training_loss = sess.run([opt, loss, learning_rate], feed_dict={X:batch_x, Y:batch_y, training:True})
        if i%10 == 0:
            test_data = manage.read_and_convert(x_test[:1000], 'test')
            result = sess.run(final_opt, feed_dict={X:test_data[:1000], training:False})
            result = [np.argmax(x) for x in result]
            print("step : %d, training loss = %g, accuracy_score = %g, learning_rate = %g" % (i, training_loss[1], metrics.accuracy_score(y_test[:1000], result), training_loss[2]))
            if(metrics.accuracy_score(y_test[:1000], result) > 0.92):
                break
                
    saver.save(sess, './data/resnet.ckpt') # 模型保存

def test():
    path = "test/"       
    manage = Datamanage()
    filelist = os.listdir(path)
    filelist.sort(key=lambda x:int(x.split('.')[0]))
    saver = tf.train.import_meta_graph("./data/resnet.ckpt.meta")
    results = []
    with tf.Session() as sess:
        saver.restore(sess, "./data/resnet.ckpt")
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("x").outputs[0]
        y = tf.get_collection("pred_network")[0]
        training = graph.get_operation_by_name("training").outputs[0]
        for i in range(len(filelist) // 100):
            s = i*100; e = (i+1)*100
            data = manage.read_and_convert(filelist[s:e], 'test')
            result = sess.run(y, feed_dict={x:data, training:False})
            result = [np.argmax(x) for x in result]
            for re in result:
                if re==0: results.append('airplane')
                elif re==1: results.append('automobile')
                elif re==2: results.append('bird')
                elif re==3: results.append('cat')
                elif re==4: results.append('deer')
                elif re==5: results.append('dog')
                elif re==6: results.append('frog')
                elif re==7: results.append('horse') 
                elif re==8: results.append('ship')
                elif re==9: results.append('truck')
            print("num=====", i*100)
        # print(results)
        manage.csv_write(results)
        print('done!!')

if __name__ == "__main__":
    train()
    # test()
