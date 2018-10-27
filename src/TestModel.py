
# coding: utf-8

# # Step 2 - Test The Model
# 
# In this notebook, we will use the model that we trained in Step 1 to drive the car around in AirSim. We will make some observations about the performance of the model, and suggest some potential experiments to improve the model.
# 
# First, let us import some libraries.

# In[ ]:


from keras.models import load_model
import sys
import numpy as np
import glob
import os

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('model/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))


# Next, we'll load the model and connect to AirSim Simulator in the Landscape environment. Please ensure that the simulator is running in a different process *before* kicking this step off.

# In[3]:


model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')


# We'll set the initial state of the car, as well as some buffers used to store the output from the model

# In[4]:


car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))


# We'll define a helper function to read a RGB image from AirSim and prepare it for consumption by the model

# In[5]:


def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    return image_rgba[76:135,0:255,0:3].astype(float)


# Finally, a control block to run the car. Because our model doesn't predict speed, we will attempt to keep the car running at a constant 5 m/s. Running the block below will cause the model to drive the car!

# In[ ]:


while (True):
    car_state = client.getCarState()
    
    if (car_state.speed < 5):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0
    
    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)
    
    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
    
    client.setCarControls(car_controls)


# ## Observations and Future Experiments
# 
# We did it! The car is driving around nicely on the road, keeping to the right side for the most part, carefully navigating all the sharp turns and instances where it could potentially go off the road. However, you would immediately notice a few other things. Firstly, the motion of the car is not smooth, especially on those bridges. Also, if you let the model running for a while (a little more than 5 minutes), you will notice that the car eventually veers off the road randomly and crashes. But that is nothing to be disheartened by! Keep in mind that we have barely scratched the surface of the possibilities here. The fact that were able to have the car learn to drive around almost perfectly using a very small dataset is something to be proud of!
# 
# > **Thought Exercise 2.1**:
# As you might have noticed, the motion of the car is not very smooth on those bridges. Can you think of a reason why it is so? Can you use one of the techniques we described in Step 0 to fix this?
# 
# > ** Thought Exercise 2.2**:
# The car seems to crash when it tries to climb one of those hills. Can you think of a reason why? How can you fix this? (Hint: You might want to take a look at what the car is seeing when it is making that ascent)
# 
# AirSim opens up a world of possibilities. There is no limit to the new things you can try as you train even more complex models and use other learning techniques. Here are a few immediate things you could try that might require modifying some of the code provided in this tutorial (including the helper files) but won't require modifying any Unreal assets.
# 
# > ** Exploratory Idea 2.1**:
# If you have a background in Machine Learning, you might have asked the question: why did we train and test in the same environment? Isn't that overfitting? Well, you can make arguments on both sides. While using the same environment for both training and testing might seem like you are overfitting to that environment, it can also be seen as drawing examples from the same probability distribution. The data used for training and testing is not the same, even though it is coming from the same distribution. So that brings us to the question: how will this model fare in a different environment, one it hasn't seen before? 
# This current model will probably not do very well, given that the other available environments are very different and contain elements that this model has never seen before (intersections, traffic, buildings etc.). But it would be unfair to ask this model to work well on those environments. Think of it like a human who has only ever driven in the mountains, never seen other cars or intersections in their entire life, is suddenly asked to drive in a city. How well do you think they will fare?
# The opposite case should be interesting though. Does training on data collected from one of the city environments generalize easily to driving in the mountains? Try it yourself to find out.
# 
# > ** Exploratory Idea 2.2**:
# We formulated this problem as a regression problem - we are predicting a continuous valued variable. Instead, we could formulate the problem as a classification problem. More specifically, we could define buckets for the steering angles (..., -0.1, -0.05, 0, 0.05, 0.1, ...), bucketize the labels, and predict the correct bucket for each image. What happens if we make this change?
# 
# > ** Exploratory Idea 2.3**:
# The model currently views a single image and a single state for each prediction. However, we have access to historical data. Can we extend the model to make predictions using the previous N images and states (e.g. given the past 3 images and past 3 states, predict the next steering angle)? (Hint: This will possibly require you to use recurrent neural network techniques)
# 
# > ** Exploratory Idea 2.4**:
# AirSim is a lot more than the dataset we provided you. For starters, we only used one camera and used it only in RGB mode. AirSim lets you collect data in depth view, segmentation view, surface normal view etc for each of the cameras available. So you can potentially have 20 different images (for 5 cameras operating in all 4 modes) for each instance (we only used 1 image here). How can combining all this information help us improve the model we just trained?
