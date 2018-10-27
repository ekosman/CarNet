import pyvjoy
import keyboard

current_x = 0x4000 # wheel
current_y = 0x4000 # accelerate
current_z = 0x4000 # brake


def register_device():
    '''
    acquire device
    '''
    j = pyvjoy.VJoyDevice(1)
    if j is None:
        return False

    # reset device
    current_x = 0x4000  # wheel
    current_y = 0x4000  # accelerate
    current_z = 0x4000  # brake

    print("Reset axis X")
    j.set_axis(pyvjoy.HID_USAGE_X, current_x)
    print("Reset axis Y")
    j.set_axis(pyvjoy.HID_USAGE_Y, current_y)
    print("Reset axis Z")
    j.set_axis(pyvjoy.HID_USAGE_Z, current_z)

    # Notice the args are (buttonID,state) whereas vJoy's native API is the other way around.
    for i in range(1, 9):  # turn all buttons off
        print("Reset button", i)
        j.set_button(i, 0)

    return j


def update_position(axis, value, device):
    if axis == 'x':
        device.set_axis(pyvjoy.HID_USAGE_X, value)
    elif axis == 'y':
        device.set_axis(pyvjoy.HID_USAGE_Y, value)
    elif axis == 'z':
        device.set_axis(pyvjoy.HID_USAGE_Z, value)


def press_button(device, button):
    device.set_button(button,1)


def release_button(device, button):
    device.set_button(button,0)


if __name__ == '__main__':
    j = register_device()
    if j is None:
        print("failed to register device")
        exit(0)

    while True:
        try:
            if keyboard.is_pressed(keyboard.KEY_DOWN):
                if current_y > 0x1:
                    current_y -= 1
                    j.set_axis(pyvjoy.HID_USAGE_Z, current_y)
            elif keyboard.is_pressed(keyboard.KEY_UP):
                if current_y < 0x8000:
                    current_y += 1
                    j.set_axis(pyvjoy.HID_USAGE_Z, current_y)
        except:
            break
