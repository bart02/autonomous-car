from helpers import constrain
from time import sleep


class Motors:
    frame_ids = {'reboot': 0}
    encoder = 0

    def __init__(self, interface=None):
        self.interface = interface
        sleep(0.5)
        self.stop()
        sleep(0.5)
        self.frame_ids['start'] = self.read('reboot')

    def go(self, angle, dir, speed):
        comm = "SPD {},{},{},{} ".format(constrain(int(angle), 65, 125), dir, speed, 0)
        # print(comm)
        if self.interface is not None:
            self.interface.write(comm.encode())
            s = self.interface.readline().replace(b'\n', b'').decode()
            if s[0] == 'F' and s[-1] == 'E':
                self.encoder = float(s[1:-1])
    command = go  # For compatibility

    def stop(self):
        self.command(90, 1, 0)

    def create_frame_id(self, frame_id):
        self.frame_ids[frame_id] = self.read('reboot')

    def create_frame_id_once(self, frame_id):
        if frame_id not in self.frame_ids:
            self.create_frame_id(frame_id)

    def read(self, frame_id='start'):
        if frame_id in self.frame_ids:
            return self.encoder - self.frame_ids[frame_id]
        else:
            return -1
