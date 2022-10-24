import pickle
import numpy as np
from fastapi import FastAPI, Request#nuovo request
from pydantic import BaseModel
#aggiunte
#https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/conditional_gan.ipynb#scrollTo=unLZou1fEQs0
from tensorflow import keras
import tensorflow as tf
from tensorflow_docs.vis import embed
import imageio
from fastapi.responses import HTMLResponse, FileResponse
#from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

latent_dim = 128
num_classes = 10

app = FastAPI(title="Conditional GAN")

#new
templates = Jinja2Templates(directory="templates")
# Represents a particular wine (or datapoint)
class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    print("loading model2")
    pickle_file = "/app/model2.pkl"#"/app/wine.pkl"
    with open(pickle_file, "rb") as file:
        global trained_gen
        trained_gen = pickle.load(file)
        print("model has been loaded.")

#ValueError: context must include a "request" key
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})
    #return "(Home) Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"

#post
#attenzione, le risposte sono cachate
@app.get("/number/{num}", response_class=HTMLResponse)
def number(request: Request, num: int):
    #print("getnum",num)
    # Choose the number of intermediate images that would be generated in
    # between the interpolation + 2 (start and last images).
    num_interpolation = 9  # @param {type:"integer"}

    # Sample noise for the interpolation.
    #print("latent_dim",latent_dim)
    interpolation_noise = tf.random.normal(shape=(1, latent_dim))
    interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
    interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))
    #print("interpolation_noise",interpolation_noise)
    def interpolate_class(first_number, second_number):
        # Convert the start and end labels to one-hot encoded vectors.
        first_label = keras.utils.to_categorical([first_number], num_classes)
        second_label = keras.utils.to_categorical([second_number], num_classes)
        first_label = tf.cast(first_label, tf.float32)
        second_label = tf.cast(second_label, tf.float32)

        # Calculate the interpolation vector between the two labels.
        percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
        percent_second_label = tf.cast(percent_second_label, tf.float32)
        interpolation_labels = (
            first_label * (1 - percent_second_label) + second_label * percent_second_label
        )

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
        fake = trained_gen.predict(noise_and_labels)
        return fake


    start_class = num
    #num#1  # @param {type:"slider", min:0, max:9, step:1}
    end_class = num
    #num#5  # @param {type:"slider", min:0, max:9, step:1}
    print(f"creo {num_interpolation} immagini del numero {num}")
    fake_images = interpolate_class(start_class, end_class)

    ###
    fake_images *= 255.0
    converted_images = fake_images.astype(np.uint8)
    converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=1)
    #embed.embed_file("animation2.gif")
    #"request": request, id; item.html
    #path = "img/apples-vs-oranges-health-benefits-1524606349.jpg"
    path = "animation.gif"
    #f"{IMAGEDIR}{files[random_index]}"
    
    # notice you can use FileResponse now because it expects a path
    return FileResponse(path)
    #return templates.TemplateResponse("item.html", {"num":num,"request": request})
"""
@app.post("/predict")
def predict(wine: Wine):
    data_point = np.array(
        [
            [
                wine.alcohol,
                wine.malic_acid,
                wine.ash,
                wine.alcalinity_of_ash,
                wine.magnesium,
                wine.total_phenols,
                wine.flavanoids,
                wine.nonflavanoid_phenols,
                wine.proanthocyanins,
                wine.color_intensity,
                wine.hue,
                wine.od280_od315_of_diluted_wines,
                wine.proline,
            ]
        ]
    )

    pred = clf.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}
"""