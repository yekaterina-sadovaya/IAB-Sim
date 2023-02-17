"""This module abstracts fragmentation of data packets
into smaller ARQ fragments"""

from gl_vars import gl
from math import ceil


class packing_params():
    """Defines parameters for packaging module"""

    def __init__(self):
        self.MTU = gl.max_codeblock_size_bytes
        # this has to be fixed or at least predictable for client relay ot work.
        self.max_retries = gl.max_ARQ_retries
        # Flag to enable CR-specific packaging logic
        self.cr_enabled = False  # ASSUME FALSE AND DELETE


class TPacketizer:
    def __init__(self, interface, params=packing_params()):
        if params is None:
            params = packing_params()
        # sequence number for ARQ packets
        self.seq = 0
        # the interface to which it belongs
        self.interface = interface
        # fragmentation/packing policy
        self.params = params
        # a buffer for all fragments that are on air
        self.frag_buf = []
        # add hooks
        interface.on_ack.append(self.ack)
        interface.on_nack.append(self.nack)
        # This is used internally
        self.__free_slots = 0
        # self.frags_in_transit=set()
        # Packet from TX queue that we are currently putting onto the air
        self.pkt = None

    def ack(self, **kwargs):
        """Fetch a new packet from TX_queue and send it to phy"""
        fragment = kwargs["fragment"]

        print("ACK on fragment")
        fragment.delivered()
        self.frag_buf.remove(fragment)

    def nack(self, **kwargs):
        fragment = kwargs["fragment"]
        # self.frags_in_transit.remove(hash(fragment))
        if fragment.retransmissions >= self.params.max_retries:
            print("DROP fragment")
            fragment.dropped()
            self.frag_buf.remove(fragment)
        else:
            print("LOST fragment")
            fragment.lost()

    @property
    def frag_buf_len(self):
        """Returns number of unsent bytes in fragmentation buffer + size
         of the packet being fragmented now, that is how many bytes should be sent to
         clear the mac packaging module assuming 100% delivery probability"""
        l = sum(f.total_bytes for f in self.frag_buf if not f.in_transit)
        if self.pkt:
            l += self.pkt.data_remaining

        return l


if __name__ == '__main__':
    class interface():
        def __init__(self):
            self.on_ack = []
            self.on_nack = []
    iface = interface()
    packetizer = TPacketizer(interface=iface)
    print(packetizer)
