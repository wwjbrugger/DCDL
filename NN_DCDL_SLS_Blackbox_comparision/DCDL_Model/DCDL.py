class DCDL:
    def __init__(self, operations):
        if not isinstance(operations, dict):
            # check if operations is a dict
            raise ValueError ('operations object has to be a dictionary but it is a {}'.format(type(operations)))

        # saves operation to perform operations should have the form {position : object}
        self.operations = operations