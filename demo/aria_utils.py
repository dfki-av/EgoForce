import subprocess
from sortedcontainers import SortedList
from collections import deque


def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)


class TimestampIndex:
    def __init__(self, max_size):
        self.max_size = max_size
        self.timestamps = SortedList()
        self.index_to_data = {}
        self.timestamp_queue = deque()  # To keep track of insertion order

    def add_timestamp(self, data, timestamp):
        # If we've reached the max size, remove the oldest timestamp
        if len(self.timestamps) >= self.max_size:
            self._remove_oldest_timestamp()

        index = self.timestamps.bisect_left(timestamp)
        self.timestamps.add(timestamp)
        self.index_to_data[index] = data
        self.timestamp_queue.append((timestamp, index))

        # Update indices for all timestamps after the inserted one
        for i in range(index + 1, len(self.timestamps)):
            old_data = self.index_to_data[i - 1]
            self.index_to_data[i] = old_data

    def _remove_oldest_timestamp(self):
        oldest_timestamp, oldest_index = self.timestamp_queue.popleft()
        self.timestamps.remove(oldest_timestamp)
        del self.index_to_data[oldest_index]

        # Update indices for all timestamps after the removed one
        for i in range(oldest_index, len(self.timestamps)):
            self.index_to_data[i] = self.index_to_data.pop(i + 1)

        # Update indices in the timestamp_queue
        self.timestamp_queue = deque(
            (ts, 
             idx - 1 if idx > oldest_index else idx
             )
            for ts, idx in self.timestamp_queue
        )

    def find_closest_data(self, reference_timestamp):
        if not self.timestamps:
            return None
        index = self.timestamps.bisect(reference_timestamp)
        if index == 0:
            return self.index_to_data[0]
        if index == len(self.timestamps):
            return self.index_to_data[index - 1]
        left = index - 1
        right = index
        if abs(self.timestamps[left] - reference_timestamp) <= abs(self.timestamps[right] - reference_timestamp):
            return self.index_to_data[left]
        else:
            return self.index_to_data[right]
