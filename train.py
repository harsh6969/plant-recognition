import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set image dimensions
img_width, img_height = 224, 224

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model
for layer in model.layers:
    layer.trainable = False

# Function to add layers on top of the base model
def add_layer_at_bottom(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

# Prepare data
train_data_dir = 'dataset/train'
val_data_dir = 'dataset/test'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=45,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                target_size=(img_height, img_width),
                                                batch_size=batch_size,
                                                class_mode='categorical')

num_classes = len(train_generator.class_indices)
FC_head = add_layer_at_bottom(model, num_classes)

# Create the final model
main_model = Model(inputs=model.input, outputs=FC_head)

# Compile the model
main_model.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

# Set callbacks
checkpoint = ModelCheckpoint("Model.h5", monitor='val_loss', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)

# Train the model
epochs = 50
history = main_model.fit(train_generator,
                         steps_per_epoch=len(train_generator),
                         epochs=epochs,
                         callbacks=[checkpoint, earlystop],
                         validation_data=val_generator,
                         validation_steps=len(val_generator))

# Optionally, save the training history
# Save your model or training history as needed
