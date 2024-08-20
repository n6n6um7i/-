import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# 建立LSTM模型
model = Sequential()

# 添加LSTM層
model.add(LSTM(units=50, return_sequences=True, input_shape=(30, 64)))  # 假設您的輸入有30個時間步和64個特徵
model.add(LSTM(units=50))  # 第二層LSTM層

# 添加全連接層
model.add(Dense(units=1, activation='sigmoid'))  # 二元分類使用sigmoid激活函數

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# 設定隨機種子以保證結果可重現
np.random.seed(42)

# 假設我們有1000個樣本，每個樣本有30個時間步，每個時間步有64個特徵
num_samples = 1000
time_steps = 30
features = 64

# 生成隨機的輸入數據 (1000, 30, 64)
x_train = np.random.random((num_samples, time_steps, features))

# 生成隨機的標籤 (0或1) 作為二元分類
y_train = np.random.randint(2, size=(num_samples, 1))

# 為驗證集生成相同格式的隨機數據
x_val = np.random.random((200, time_steps, features))
y_val = np.random.randint(2, size=(200, 1))

print("隨機數據生成完畢")

# 訓練模型並保存最佳模型
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=8)

# 建立TensorFlow Lite轉換器
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#converter._experimental_lower_tensor_list_ops = False

# 轉換模型為TensorFlow Lite格式
tflite_model = converter.convert()

# 保存TensorFlow Lite模型
with open('F:\mmwave_radar\lstmmodel.tflite', 'wb') as f:
    f.write(tflite_model)

print("模型已保存為 model.tflite")
