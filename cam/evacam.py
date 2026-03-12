class EVACAMConfig:
    def match(self, row, query) -> tuple[bool, float, float]:
        """Returns (matched, energy, latency)"""
        raise NotImplementedError

    def write(self, data) -> tuple:
        """Returns (data, energy, latency) where data may include variation effects"""
        raise NotImplementedError
