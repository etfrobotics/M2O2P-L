import Leap, sys, time, ctypes, numpy
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import socket
import uuid
from datetime import datetime
from _packages.HTTP.BasicHttpAgent import *
import requests

class SampleListener(Leap.Listener):
    def __init__(self):
        super().__init__()
        self.task_uuid = 0
        self.task = 1
        self.step = 1

    def on_init(self, controller):
        global flint_client
        print("Initialized")
        self.model = self.create_vgg16()
        self.model.load_weights('model_vgg_new.h5')


    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        global flint_client
        global i
        frame = controller.frame()
        
        if frame.id % 1 == 0:
            hands = frame.hands
            numHands = len(hands)
            #print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d" % (
            #   frame.id, frame.timestamp, numHands, len(frame.fingers), len(frame.tools)))
            if numHands > 0 and len(frame.fingers) > 0:
                image = self.get_image(controller)
                simbol = self.test_image(image)
                print(simbol)
                #Based on application code below can be changed to send appropriate msg to FIWARE
                if simbol == 'Open hand' or simbol == 'L symbol' or simbol == 'Fist' or simbol == 'Vertical Hand': 

                    with open('msg_json_next_step.json') as f_ns_json:
                        ns_data = json.loads(f_ns_json.read())
                    ns_data["id"] = ns_data["id"] + str(uuid.uuid4())
                    now = datetime.now()
                    ns_data["status"]["observedAt"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
                    ns_data["status"]["value"] = "started"
                    #Put here IP of your FIWARE ORION Context Broker
                    requests.post("http://127.0.0.1:1026/ngsi-ld/v1/entities/",json=ns_data,headers={"Content-Type":"application/ld+json"})
                #Show IMG
                #img = Image.fromarray(image).convert("RGB")
                #title_text = simbol
                #title_font = ImageFont.truetype('Roboto-Black.ttf', 50)
                #image_editable = ImageDraw.Draw(img)
                #image_editable.text((15,15), title_text,(237, 230, 211),font=title_font)
                #img.show()
                #img.save("M2O2P-L"+simbol+".png")
                time.sleep(2)
                #i+=1

            

    def create_vgg16(self):
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=(120, 320, 3),  # Shape of our images
                                                       include_top=False,  # Leave out the last fully connected layer
                                                       weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        x = tf.keras.layers.Flatten()(base_model.output)

        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = tf.keras.layers.Dense(512, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

        # Add a dropout rate of 0.5
        x = tf.keras.layers.Dropout(0.2)(x)

        # Add a final sigmoid layer with 1 node for classification output
        x = tf.keras.layers.Dense(4, activation='sigmoid',
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

        model = tf.keras.models.Model(base_model.input, x)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy',
                      metrics=['acc'])

        return model

    def get_image(self, controller):
        image_list = controller.frame().images
        left_image = image_list[0]
        right_image = image_list[1]
        image_buffer_ptr = left_image.data_pointer
        ctype_array_def = ctypes.c_ubyte * left_image.width * left_image.height
        as_ctype_array = ctype_array_def.from_address(int(image_buffer_ptr))
        image_numpy_first = numpy.ctypeslib.as_array(as_ctype_array)

        return image_numpy_first

    def test_image(self, image):
        img = Image.fromarray(image).convert('L')
        img1 = img.copy()
        img = img.resize((320, 120))
        arr = numpy.array(img)
        arr = numpy.tile(arr[:, :, None], [1, 1, 3])
        arr = arr / 255
        arr = numpy.expand_dims(arr, axis=0)

        start_time = time.time()
        b = self.model.predict(arr)
        end_time = time.time()

        action = numpy.argmax(b)
        vector = ['Fist', 'Open hand', 'L symbol', 'Vertical Hand']

        return (vector[action])

def agent_task(flint_agent):
    flint_agent.run()

def main():
    listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy(Leap.Controller.POLICY_IMAGES)
    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass

    # Remove the sample listener when done
    controller.remove_listener(listener)

if __name__ == "__main__":
    main()
