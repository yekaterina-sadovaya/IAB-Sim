"""THis module abstracts fragmentation of data packets
into smaller ARQ fragments. Those are not exactly MAC PDU's,
which in general can consist of multiple ARQ blocks. The point here is
to provide a simple model, not an exact one."""

from gl_vars import gl
import packet01
from math import ceil


class packing_params():
    """Defines parameters for packaging module"""

    def __init__(self):
        self.MTU = gl.MAX_ARQ_BLOCK
        # this has to be fixed or at least predictable for client relay ot work.
        self.max_retries = gl.MAX_ARQ_RETRIES
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

    def arrange_burst(self, max_slots):
        """This should be called when the size of burst and AMC params
        are announced by the BS scheduler"""
        print("Arranging burst of " + str(max_slots) + " maximum size")
        # assert max_slots > 0, "Scheduler bug- zero-sized allocation"
        # if self.frags_in_transit:
        #    debug(self.frags_in_transit)
        slot_bits = self.interface.amc.slots2bits(1)
        tx_pow = self.interface.amc.tx_power
        bler_curve = self.interface.amc.amc_code.f
        repetitions = self.interface.amc.repetitions
        assert slot_bits > 0, "Somehow we can not send any data - terminating!"
        free_slots = max_slots
        # the buffer for fragments to be transmitted in this burst
        txbuf = []
        # Fill the burst with retransmissions
        self.frag_buf.sort(key=id)
        for f in self.frag_buf:
            if f.in_transit:
                # Check for timeouts on ACK/NACK from BS
                if gl.sched.time - f.time_sent_to_phy > LTE.FRAME_TICKS:
                    f.dropped()
                    self.frag_buf.remove(f)
                # debug("Skipping packet that is still in transit")
                continue
            f.size_slots = int(ceil(f.total_bytes * 8.0 / slot_bits))
            print("Considering fragment " + str(f))

            if f.size_slots > free_slots:
                print("Refragmenting fragment " + str(f))
                f_new = f.split(int(free_slots * slot_bits / 8.0), free_slots, self.seq)
                if f_new is None:
                    break
                self.seq += 1
                # put the new fragment into frag_buf
                self.frag_buf.append(f_new)
                f = f_new
                # Nothing can fit in whatever remains
                free_slots = 0
            else:
                free_slots -= f.size_slots
            print("Retransmitting fragment " + str(f))
            f.tx_power = tx_pow - 2 if self.params.cr_enabled else tx_pow
            f.bler_curve = bler_curve
            f.sent_to_phy()
            txbuf.append(f)
            if free_slots == 0:
                # Break if we have exhausted the resource available
                break
            assert free_slots > 0, "Make sure we do not overuse resource"
        print("Assigned " + str(max_slots - free_slots) + " slots to retransmissions")
        # align MTU to slot boundary
        MTU = self.params.MTU * 8
        q = self.interface.tx_queue
        while (free_slots > 0) and (self.pkt or q):
            if not self.pkt:
                self.pkt = q.pop()
            print("Using data packet " + str(self.pkt))
            size = self.pkt.data_remaining
            # figure what is the largest possible fragment we should make
            frag_size = min(MTU, size) + gl.MAC_OVERHEAD
            # See how much it takes in terms of slots
            frag_slots = int(ceil(frag_size * 8.0 / slot_bits))
            # and check if it fits into transmit grant
            if frag_slots > free_slots:
                frag_slots = free_slots
                # see if the fragment can be expanded to occupy entire symbol
            frag_size = int(slot_bits / 8.0 * frag_slots) - gl.MAC_OVERHEAD
            if frag_size > size:
                frag_size = size
            if frag_size < 1:
                frag_size = 1
            frag = packet01.TARQFragment(data_bytes=frag_size,
                                         oh_bytes=gl.MAC_OVERHEAD,
                                         size_slots=frag_slots,
                                         source_sdu=self.pkt,
                                         source_iface=self.interface,
                                         seq=self.seq)
            if self.pkt.onair:
                # we can as well notify everyone else it is gone
                self.interface.event_TX_packet_departure(pkt=self.pkt)
                self.pkt = None
            self.seq += 1
            self.frag_buf.append(frag)
            frag.sent_to_phy()
            frag.tx_power = tx_pow
            frag.bler_curve = bler_curve
            frag.repetitions = repetitions
            txbuf.append(frag)
            free_slots -= frag_slots

        if not txbuf:
            print("Empty burst formed, most likely a bug")
        return txbuf


if __name__ == '__main__':
    class interface():
        def __init__(self):
            self.on_ack = []
            self.on_nack = []
    iface = interface()
    packetizer = TPacketizer(interface=iface)
    print(packetizer)