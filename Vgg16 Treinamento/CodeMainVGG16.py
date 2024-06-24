import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Função para redimensionar as imagens
def resize_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

# Caminhos para as pastas com as imagens originais
input_folder_cat = 'Imagens'
input_folder_not_cat = 'ImagensNaoGato'

# Caminhos para as pastas onde as imagens redimensionadas serão salvas
output_folder_cat = 'ImagensValidadas/gato'
output_folder_not_cat = 'ImagensValidadas/naogato'

# Redimensionar as imagens
resize_images(input_folder_cat, output_folder_cat)
resize_images(input_folder_not_cat, output_folder_not_cat)

# Carregar o modelo VGG16 pré-treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas para a nova tarefa
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Geradores de dados para treino e validação
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'ImagensTreino',  # Caminho dos seus dados de treino
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'ImagensTreino',  # Caminho dos seus dados de validação
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Avaliar o modelo
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'ImagensTeste',  # Caminho dos seus dados de teste
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

evaluation = model.evaluate(test_generator)
print(f"Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")

# Função para carregar e pré-processar uma imagem
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Função para fazer predições com o novo modelo
def predict_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    return "Gato" if prediction[0] < 0.5 else "Não Gato"

# Listar todas as imagens redimensionadas na pasta de teste
img_files = []
for category in ['gato', 'naogato']:
    category_path = os.path.join('ImagensTeste', category)
    img_files.extend([(os.path.join(category_path, f), category) for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

# Processar e exibir cada imagem com suas predições
for img_path, category in img_files:
    prediction = predict_image(model, img_path)

    # Exibir as predições
    print(f"Prediction for {os.path.basename(img_path)} ({category}): {prediction}\n")

    # Carregar e exibir a imagem
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{prediction} ({category})")
    plt.show()
