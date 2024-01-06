import tensorflow as tf
import pandas as pd

DATASET_PATH = "/Users/vito/Desktop/sign_mnist_train/sign_mnist_train.csv"
df = pd.read_csv(DATASET_PATH)

images = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
labels = df['label'].values.astype('int32')

dataset = tf.data.Dataset.from_tensor_slices((images, labels))

batch_size = 32
train_dataset = dataset.map(lambda x, y: (tf.image.resize(x, (28, 28))/255.0, y)).shuffle(buffer_size=10000).batch(batch_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output con softmax per classificazione multiclasse
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=1)

