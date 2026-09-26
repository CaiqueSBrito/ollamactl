class OllamaError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code  # HTTP status, when the server answered with an error
