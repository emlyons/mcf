from mcf.data_store.data_store_status import DataStoreStatus

class DataStore:

    def __init__(self):
        self.data = {}

    def put(self, key, value) -> DataStoreStatus:
        status = DataStoreStatus.SUCCESS

        if key in self.data:
            status = DataStoreStatus.ERROR_KEY_COLLISION
            
        if status == DataStoreStatus.SUCCESS:
            self.data[key] = value
            
        return status


    def get(self, key) -> tuple[DataStoreStatus, object]:
        status = DataStoreStatus.SUCCESS
        value = None

        if key not in self.data:
            status = DataStoreStatus.ERROR_INVALID_KEY

        if status == DataStoreStatus.SUCCESS:
            value = self.data[key]

        return status, value
    
    def remove(self, key):
        status = DataStoreStatus.SUCCESS
        
        if key not in self.data:
            status = DataStoreStatus.ERROR_INVALID_KEY

        if status == DataStoreStatus.SUCCESS:
            self.data.pop(key)

        return status

    def __len__(self):
        return len(self.data)
    