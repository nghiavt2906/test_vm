import numpy as np
from npsocket import SocketNumpyArray

sock_receiver = SocketNumpyArray()
sock_receiver.initalize_receiver(9999)
while True:
    frame = sock_receiver.receive_array()  
    print(np.asarray(frame).shape)

    # sock_receiver.send_numpy_array(np.array(['Vo Trong Nghia']))
    # res = [['Vo Trong Nghia', [[1, 2], [3, 4]]], ['John', [[3, 4], [8, 10]]]]
    sock_receiver.response_to_sender(res)