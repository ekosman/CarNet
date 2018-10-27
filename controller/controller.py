import pyvjoy

current_x = 0x4000 # wheel
current_y = 0x4000 # accelerate
current_z = 0x4000 # brake


def register_device():
    #acquire device
    j = pyvjoy.VJoyDevice(1)
    if j is None:
        return False

    # reset device


    return j


#Pythonic API, item-at-a-time
current_pos = 0x4000


#turn button number 15 on
j.set_button(15,1)

#Notice the args are (buttonID,state) whereas vJoy's native API is the other way around.


#Set X axis
j.set_axis(pyvjoy.HID_USAGE_X, current_pos)

import keyboard #Using module keyboard
while True:#making a loop
    try: #used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed(keyboard.KEY_DOWN):#if key 'q' is pressed
            if current_pos > 0x1:
                current_pos -= 1
                j.set_axis(pyvjoy.HID_USAGE_X, current_pos)
        elif keyboard.is_pressed(keyboard.KEY_UP):
            if current_pos < 0x8000:
                current_pos += 1
                j.set_axis(pyvjoy.HID_USAGE_X, current_pos)
    except:
        break #if user pressed a key other than the given key the loop will break