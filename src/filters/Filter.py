class Filter:
    def __init__(self):
        """
        Initializes the Filter object.

        Attributes:
            calls_counter (int): Counter to track the number of calls to the filter.
            cache (list): List to manually store some of the results.
            log (str): Message that is printed before calling an operation.

        Returns:
            None
        """
        self.calls_counter: int = 0
        self.cache: list = []
        self.log = "Default call message"

    def start_log(self):
        """
        Prints the log message.

        Returns:
            str: The log message defined for the operation (or the default).
        """
        print(self.log)