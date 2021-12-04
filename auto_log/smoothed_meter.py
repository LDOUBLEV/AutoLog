class SmoothedMeter(object):
    def __init__(self, key, unit=""):
        # basic info
        self.key = key
        self.unit = unit
        # value and count for local average
        self.local_sum = 0
        self.local_count = 0
        # value and count in all for global average
        self.global_sum = 0
        self.global_count = 0
        self.fmt = None

    def reset(self, scope="global"):
        self.local_sum = 0
        self.local_count = 0

        if scope == "global":
            self.global_sum = 0
            self.global_count = 0
        return

    def update(self, value, n=1):
        assert isinstance(value, (
            int,
            float)), "value can just be int or float bu got {}".format(value)
        if self.fmt is None:
            self.fmt = "{} {:.7f}" if isinstance(value, float) else "{} {}"
        self.local_count += n
        self.local_sum += value * n

        self.global_sum += value * n
        self.global_count += n

    def _average(self, scope="avg"):
        assert self.fmt is not None, "update must be used before add_record"
        if scope == "avg":
            if self.local_count <= 0:
                return ""
            else:
                return self.fmt.format(self.key,
                                       self.local_sum / self.local_count)
        else:
            if self.global_count <= 0:
                return ""
            else:
                return self.fmt.format(self.key,
                                       self.global_sum / self.global_count)

    def get_sum(self, scope="avg"):
        if scope == "avg":
            return self.local_sum
        else:
            return self.global_sum

    def log(self, scope="avg"):
        info = self._average(scope=scope)
        return info
