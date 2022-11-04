# imoprt之後要用到的librarys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# batch size參數
batch_size = 256
# 影像shape
img_height = 28
img_width = 28

# 資料路徑(由1_prepareData.py處理完成的資料)
train_dir = './train_image'
test_dir = './test_image'

# 準備training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, # train data路徑
    color_mode='grayscale', # 灰階
    seed=666, # 亂數seed
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 準備testing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, # test data路徑
    color_mode='grayscale', # 灰階
    seed=666, # 亂數seed
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 檢查class names
class_names = train_ds.class_names
print(class_names)

# 檢查data shape
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# 建模型(以下激活函數都為relu)
model = models.Sequential()
# 捲積層3x3
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 池化層2x2
model.add(layers.MaxPooling2D((2, 2)))
# 捲積層3x3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# 池化層2x2
model.add(layers.MaxPooling2D((2, 2)))
# 捲積層3x3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# 全連接層
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 開始訓練(30個epochs)
history = model.fit(train_ds, epochs=30, validation_data=test_ds)

# 繪製accuracy歷史
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('accuracy.png')
plt.close()

# 繪製loss歷史
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label = 'test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('loss.png')
plt.close()