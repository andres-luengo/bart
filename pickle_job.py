import logging

# does everything caleb's code does minus the managerial nonsense
class PickleJob:
    def __init__(self, batch: list[list[str]]):
        self._batch = batch

        self._logger = logging.getLogger(__name__)
    
    def run(self):
        pass