class Filter:

    def __init__(self):
        """
        Handmade caching parameters for optimizing Filters that are called many times
        """
        self.calls_counter: int = 0
        self.cache: list = []
