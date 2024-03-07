class Filter:

    def __init__(self):
        """
        Handmade caching parameters for optimizing Filters that are called many times.
        cache - for manually storing some of the results.
        log - a message that is called before calling an operation.
        """
        self.calls_counter: int = 0
        self.cache: list = []
        self.log = "Default call message"

    def start_log(self):
        """
        A method that is inherited and can be called for every operation.

        :return: the log we redefined for the operation (or the default)
        """
        print(self.log)
