import os
import glob
import json
import cv2
import gradio as gr

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# from keras.utils import image_utils
from keras.preprocessing import image
from model import GoogLeNet

flower_category = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = GoogLeNet(class_num=5, aux_logits=False)
# weights_path = "C:/桌面/college/大四上/project/flower_GoogleNet/save_weights/myGoogLeNet.ckpt"
weights_path = "C:/桌面/college/大四上/project/flower_GoogleNet/save_weights/myGoogLeNet.ckpt3"
# weights_path = "C:/桌面/college/大四上/project/flower_GoogleNet/save_weights/myGoogLeNet.ckpt3"
assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
model.load_weights(weights_path)

#draw confusion matrix
    # plt.figure(figsize=(10, 10))
    # plt.imshow(result2, cmap=plt.cm.Greens)
    # plt.xticks(range(5), flower_category, rotation=45)
    # plt.yticks(range(5), flower_category)
    # plt.colorbar()
    # plt.xlabel('True label')
    # plt.ylabel('Predicted label')
    # plt.show()

def main():
    im_height = 224
    im_width = 224

    # load image
    img_path = "C:/桌面/college/大四上/project/flower_dataset/Internet_test/daisy.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # resize image to 224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # scaling pixel value and normalize
    img = ((np.array(img) / 255.) - 0.5) / 0.5

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model.summary()
    # model.load_weights("./save_weights/myGoogLenet.h5", by_name=True)  # h5 format

    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i]))
    plt.show()


def predict_flower(test_image):
    # resize the test_image
    test_image = cv2.resize(test_image, (224, 224))
    plt.imshow(test_image)
    # convert the image to array
    test_image = image.img_to_array(test_image)
    # test_image = image_utils.img_to_array(test_image)
    # expand the dimensions
    test_image = np.expand_dims(test_image, axis=0)
    # predict the image
    result2 = model.predict(test_image)
    result = np.squeeze(model.predict(test_image))
    predict_class= np.argmax(result)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i]))
    plt.show()


    # get the index of the max value
    #predict_class = np.argmax(result)
    # return the flower category

    # 该图片为每种花的品种的可能性
    # resultAll = np.squeeze(result)
    # for i in range(len(resultAll)):
    #     print(resultAll[i])

    return flower_category[predict_class]


if __name__ == "__main__":
    gr.Interface(fn=predict_flower, inputs="image", outputs="label").launch()


