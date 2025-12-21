# ------------------------------------------------------------------------------
#  Copyright (c) 2025 Efe Deniz Asan <asan.efe.deniz@gmail.com>
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  Efe Deniz Asan. The intellectual and technical concepts contained herein
#  are proprietary to Efe Deniz Asan and are protected by trade secret or
#  copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained
#  from Efe Deniz Asan or via email at <asan.efe.deniz@gmail.com>.
# ------------------------------------------------------------------------------

import pyaudio

def list_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("\nAvailable Audio Input Devices:")
    print("------------------------------")
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            print(f"ID {i}: {name}")
    print("------------------------------\n")
    p.terminate()

if __name__ == "__main__":
    list_devices()
