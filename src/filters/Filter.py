class Filter:

    def __init__(self):
        """
        Handmade caching parameters for optimizing Filters that are called many times and flag to return all results.
        Default = true because many Filters return only one result.
        """
        self.calls_counter: int = 0
        self.cache: list = []
        self.return_all = True
