import tensorflow as tf
import core.model
import core.util
import os

graph = core.model.ModelPredict()

for file in os.listdir("valid/"):
    print(file)
    print(core.util.predict(graph=graph,image_in='valid/'+file))
