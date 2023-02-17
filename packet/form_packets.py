import itertools
from gl_vars import gl
from stat_container import st


class PackingParams:
    """Defines parameters for packaging module"""

    def __init__(self):
        self.MTU = gl.MAX_ARQ_BLOCK
        # this has to be fixed or at least predictable for client relay ot work.
        self.max_retries = gl.max_ARQ_retries
        # Flag to enable CR-specific packaging logic
        self.cr_enabled = False  # ASSUME FALSE AND DELETE


class TDataPacket:
    """ Packet class (the ones we are delivering) """
    _seq = itertools.count()

    def __init__(self, source: int, destination: int, size_bytes: int, seq: int = None, current_hop: int = None,
                 label=None):
        """
        :param size_bytes: size of the packet
        :param source: source node (used for statistics)
        :param destination: identifier of the destination
        :param label: routing label (if necessary/applicable)
        :param seq: sequence number (to ensure uniqueness)
        """

        # Sequence number (unique for a node TX/RX direction)
        if seq is None:
            self.seq = next(self._seq)
        else:
            self.seq = seq

        # Size in bytes
        assert size_bytes > 0, "Zero-sized packets are not allowed!"
        self.size_bytes = size_bytes
        # Source node
        self.source = source
        # Destination node ID (int)
        self.destination = destination

        # Flow label (if used identifies a connection if multiple ones are present)
        self.label: int = label

        # Time to live in hops
        self.TTL: int = 64

        # Hops counter
        self.hops_number = 0

        # Next node id (if needed to specify intermediate node for multipoint interfaces)

        # Current node number
        self.current_hop: int = current_hop

        # Time of arrival (to the layer, i.e. it arrives later to MAC layer)
        self.arrival_time = st.simulation_time_s
        self.service_enter_time = None

        self.first_in_a_burst = False
        self.last_in_a_burst = False

