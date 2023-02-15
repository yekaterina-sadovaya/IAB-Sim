from gl_vars import gl
from stat_container import st

from math import floor


class packing_params():
    """Defines parameters for packaging module"""

    def __init__(self):
        # self.MTU = gl.MAX_ARQ_BLOCK
        # this has to be fixed or at least predictable
        self.max_retries = gl.max_ARQ_retries


class ARQ_block:
    def __init__(self, current_slot, params=packing_params()):
        if params is None:
            params = packing_params()
        # fragmentation/packing policy
        self.params = params
        self.packets = []
        # number of times block was retransmitted
        self.times_retransmitted = 0
        # HARQ process ID
        self.ARQ_ID = self.calc_process_ID(current_slot)
        # If a block was received correctly
        self.correctness_flag = None
        self.size_bits = None

    def calc_process_ID(self, current_slot):
        return floor(current_slot*10)

    def add_packet(self, packet_ID):
        self.packets.append(packet_ID)

    def ack(self, fragment, interface):
        interface.frag_buf.remove(fragment)

    def nack(self, fragment):
        if fragment.retransmissions >= self.params.max_retries:
            print("DROP fragment")
            st.segments_buff.remove(fragment)
        else:
            fragment.retransmissions += 1


