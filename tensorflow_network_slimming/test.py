"""
Based on the following code
https://github.com/Woshiwzl1997/Network-Slimming-Using-Keras
"""
import os
import copy
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class DataMNIST:
    def __init__(self):
        (x_train, y_train), \
            (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        self.x_train = x_train.astype(np.float32)[..., None] / 255
        self.x_test = x_test.astype(np.float32)[..., None] / 255
        self.ys_train = y_train
        self.ys_test = y_test
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

class MyModel(tf.keras.Model):
    def __init__(self, l1_factor, num_ch=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if num_ch is None:
            num_ch = [64, 64, 64, 64]

        self.conv0 = tf.keras.layers.Conv2D(num_ch[0], (3, 3), (1, 1), "SAME",
                                            name="conv0")
        self.conv1 = tf.keras.layers.Conv2D(num_ch[1], (3, 3), (1, 1), "VALID",
                                            name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(num_ch[2], (3, 3), (1, 1), "VALID",
                                            name="conv2")
        self.dense3 = tf.keras.layers.Dense(num_ch[3], name="dense3")
        self.dense4 = tf.keras.layers.Dense(10, name="dense4")

        self.bn0 = tf.keras.layers.BatchNormalization(
            gamma_regularizer=tf.keras.regularizers.l1(l1_factor), name="bn0")
        self.bn1 = tf.keras.layers.BatchNormalization(
            gamma_regularizer=tf.keras.regularizers.l1(l1_factor), name="bn1")
        self.bn2 = tf.keras.layers.BatchNormalization(
            gamma_regularizer=tf.keras.regularizers.l1(l1_factor), name="bn2")
        self.bn3 = tf.keras.layers.BatchNormalization(
            gamma_regularizer=tf.keras.regularizers.l1(l1_factor), name="bn3")

        self.tmp_flatten = None
        self.tmp_dense3 = None

    def call(self, inputs):
        x = inputs

        x = self.conv0(x)
        x = self.bn0(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Flatten()(x)

        self.tmp_flatten = x

        x = self.dense3(x)

        self.tmp_dense3 = x
        
        x = self.bn3(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = self.dense4(x)
        x = tf.keras.layers.Activation("softmax")(x)

        y = x
        return y

class TrainParam:
    def __init__(self, init_lr=1e-3, epochs=500, batch_size=100,
                 validation_split=0.1, verbose=0, lr_schedule=True,
                 pruning=False):
        self.init_lr = init_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose

def train_model(data, model, train_param, dname_logs):
    """
    Train model and save logs to dname_logs
    """

    optimizer = tf.keras.optimizers.Adam(train_param.init_lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

    patience = 15
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience,
                                                  restore_best_weights=True),
                 tf.keras.callbacks.TensorBoard(log_dir=dname_logs,
                                                histogram_freq=1)]

    history = model.fit(data.x_train, data.y_train,
                        batch_size=train_param.batch_size,
                        epochs=train_param.epochs,
                        validation_split=train_param.validation_split,
                        callbacks=callbacks, verbose=train_param.verbose)

    loss, acc = model.evaluate(data.x_test, data.y_test, verbose=0)

    return loss, acc, history.history

def save_results(model, test_loss, test_acc, history, dname_results,
                 dname_weights, test_name):

    os.makedirs(dname_results, exist_ok=True)
    os.makedirs(dname_weights, exist_ok=True)
    
    col_label = ["test_loss", "test_acc"]
    df = pd.DataFrame(np.array([[test_loss, test_acc]]), columns=col_label)
    df.to_csv(os.path.join(dname_results, test_name + f"_metrics.csv"))

    col_label = list(history.keys())
    df = pd.DataFrame(np.array(list(history.values())).T, columns=col_label)
    df.to_csv(os.path.join(dname_results, test_name + f"_history.csv"))

    fig = plt.figure(figsize=(12.8, 9.6))
    plt.subplot(211)
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.grid(), plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Loss")
    plt.subplot(212)
    plt.plot(history["accuracy"], label="accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.grid(), plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Accuracy")
    plt.savefig(os.path.join(dname_results, test_name + f"_history.png"))
    plt.close()
    #plt.show()

    model.save_weights(os.path.join(dname_weights, test_name + f"_weights.h5"))

"""
根据剪枝比例，遍历整个网络，统计每层需要剪去的通道,并将其置零
cap：channel after pruning
"""
def freeze_build_cap(model,percent):
    """
    :param model: 模型
    :param percent: 剪枝比例
    :return: 每层剪枝过后的通道：cap
    """
    total=0#整个网络的特征数目之和
    for m in model.layers:
        if isinstance(m,tf.keras.layers.BatchNormalization):#如果发现BN层
            total += m.get_weights()[0].shape[0]#BN.get_wrights():获得0：gamma,1：beta,2:moving_mean,3:moving_variance

    bn=np.zeros(total)
    index=0
    for m in model.layers:
        if isinstance(m, tf.keras.layers.BatchNormalization):
            size=m.get_weights()[0].shape[0]
            bn[index:(index+size)]=np.abs(copy.deepcopy(m.get_weights()[0]))# 把所有BN层gamma值拷贝下来
            index+=size

    #根据所有BN层的权重确定剪枝比例
    y=np.sort(bn)#将网络所有BN层的权重排序
    thre_index=int(total*percent)#确顶要保留的参数大小
    thre=y[thre_index]#最小的权重值

    pruned=np.array(0.0)
    cap=[]#确定每个BN层要保留的参数数目
    cap_mask=[]#每个BN层要保留的参数MASK
    for k,m in enumerate(model.layers):
        if isinstance(m,tf.keras.layers.BatchNormalization):
            weight_copy=np.abs(copy.deepcopy(m.get_weights()[0]))
            mask=np.array([1.0 if item>thre else 0.0 for item in weight_copy])# 小于权重thre的为0，大于的为1,唯独保持不变
            pruned=pruned + mask.shape[0] -np.sum(mask)# 小于权重thre的为0，大于的为1,唯独保持不变
            m.set_weights([                   #貌似只能一起赋值
                m.get_weights()[0] * mask,
                m.get_weights()[1] * mask,
                m.get_weights()[2],
                m.get_weights()[3]
            ])
            cap.append(int(np.sum(mask)))
            cap_mask.append(copy.deepcopy(mask))
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(np.sum(mask))))
    print('Pre-processing Successful!')

    print("Num of layer after pruning):")
    print(cap)
    return cap, cap_mask

def set_compact_model_weights(base_model, pruned_model, cap_mask):
    bn_id_in_cap = 0
    cov_id_in_cap = 0
    for o, c in zip(base_model.layers, pruned_model.layers):
        if isinstance(o, tf.keras.layers.BatchNormalization):
            idx = np.squeeze(np.argwhere(cap_mask[bn_id_in_cap] != 0))
            gamma=o.get_weights()[0][idx]  # gamma
            beta=o.get_weights()[1][idx] # beta
            mean=o.get_weights()[2][idx]  # mean
            var=o.get_weights()[3][idx]
            if idx.size == 1: #维度必须为一维
                gamma = np.expand_dims(gamma, 0)
                beta = np.expand_dims(beta, 0)
                mean = np.expand_dims(mean, 0)
                var = np.expand_dims(var, 0)
            c.set_weights([copy.deepcopy(gamma),#gamma
                           copy.deepcopy(beta),#beta
                           copy.deepcopy(mean),#mean
                           copy.deepcopy(var)])#var
            bn_id_in_cap +=1

        elif isinstance(o, tf.keras.layers.Conv2D):
            if cov_id_in_cap == 0:
                idx_out=np.squeeze(np.argwhere(cap_mask[cov_id_in_cap] != 0))

                w=o.get_weights()[0][:,:,:,idx_out]
                if idx_out.size==1:
                    w=np.expand_dims(w,3)

                b = o.get_weights()[1][idx_out]
                if idx_out.size == 1:
                    b = np.expand_dims(b, 0)

                c.set_weights([copy.deepcopy(w),  # conv weight
                               copy.deepcopy(b)  # bias
                ])

            elif cov_id_in_cap == 1 or cov_id_in_cap == 2:
                idx_in = np.squeeze(np.argwhere(cap_mask[(cov_id_in_cap - 1)] != 0))
                idx_out = np.squeeze(np.argwhere(cap_mask[cov_id_in_cap] != 0))

                w = o.get_weights()[0][:, :, :, idx_out]
                if idx_out.size == 1:
                    w = np.expand_dims(w, 3)

                w = w[:, :, idx_in, :]
                if idx_in.size == 1:
                    w = np.expand_dims(w, 2)

                b = o.get_weights()[1][idx_out]
                if idx_out.size == 1:
                    b = np.expand_dims(b, 0)

                c.set_weights([copy.deepcopy(w),  # conv weight
                               copy.deepcopy(b)  # bias
                ])
            
            cov_id_in_cap += 1

        elif isinstance(o, tf.keras.layers.Dense):
            if cov_id_in_cap == 3:
                idx_in = np.squeeze(np.argwhere(cap_mask[(cov_id_in_cap - 1)] != 0))
                idx_out = np.squeeze(np.argwhere(cap_mask[cov_id_in_cap] != 0))

                w = o.get_weights()[0][:, idx_out]
                if idx_in.size == 1:
                    w = np.expand_dims(w, 1)
                
                w = w.reshape(2, 2, -1, w.shape[-1])
                w = w[:, :, idx_in, :]
                w = w.reshape(-1, w.shape[-1])

                b = o.get_weights()[1][idx_out]
                if idx_out.size == 1:
                    b = np.expand_dims(b, 0)

                c.set_weights([copy.deepcopy(w),  # conv weight
                               copy.deepcopy(b)  # bias
                ])

            elif cov_id_in_cap == 4:
                idx_in = np.squeeze(np.argwhere(cap_mask[(cov_id_in_cap - 1)] != 0))

                w = o.get_weights()[0][idx_in, :]
                if idx_in.size == 1:
                    w = np.expand_dims(w, 0)

                b = o.get_weights()[1]

                c.set_weights([copy.deepcopy(w),  # conv weight
                               copy.deepcopy(b)  # bias
                ])

            cov_id_in_cap += 1

data = DataMNIST()

l1_factor = 1e-2

base_model = MyModel(l1_factor)
base_model(data.x_train[:2])

dname_data = "data"
base_name = "base"

train_base = True
#train_base = False
if train_base:
    train_param = TrainParam()
    loss, acc, hist = train_model(data, base_model, train_param, "logs")
    save_results(base_model, loss, acc, hist, "results", "data", base_name)
else:
    base_model.load_weights(os.path.join(dname_data, base_name + "_weights.h5"))
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    base_model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

loss, acc = base_model.evaluate(data.x_test, data.y_test)
print(f"Base: loss={loss} acc={acc}")

# print(base_model.bn0.get_weights()[0])
# print(base_model.bn1.get_weights()[0])
# print(base_model.bn2.get_weights()[0])
# print(base_model.bn3.get_weights()[0])

pruning_ratio = 0.9
cap, cap_mask = freeze_build_cap(base_model, pruning_ratio)

pruned_model = MyModel(0, cap)
pruned_model(data.x_train[:2])

set_compact_model_weights(base_model, pruned_model, cap_mask)

optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
pruned_model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

# print(base_model(data.x_train[:2]))
# print(pruned_model(data.x_train[:2]))
                           
loss, acc = pruned_model.evaluate(data.x_test, data.y_test)
print(f"Pruned: loss={loss} acc={acc}")

finetuned_name = "finetuned"

train_param = TrainParam()
loss, acc, hist = train_model(data, pruned_model, train_param, "logs")
save_results(pruned_model, loss, acc, hist, "results", "data", finetuned_name)

print(f"Fine-tuned: loss={loss} acc={acc}")
