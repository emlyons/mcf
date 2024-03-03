from mcf.data_store.data_store_status import DataStoreStatus

class DataStore:

    def __init__(self):
        self.data = {}

    def put(self, key, field, value) -> DataStoreStatus:
        status = DataStoreStatus.SUCCESS

        if key not in self.data:
            self.data[key] = {}
        
        if field in self.data[key]:
            status = DataStoreStatus.ERROR_FIELD_COLLISION
            
        if status == DataStoreStatus.SUCCESS:
            self.data[key][field] = value
            
        return status


    def get(self, key, field) -> tuple[DataStoreStatus, object]:
        status = DataStoreStatus.SUCCESS
        value = None

        if key not in self.data:
            status = DataStoreStatus.ERROR_INVALID_KEY

        if status == DataStoreStatus.SUCCESS and field not in self.data[key]:
            status = DataStoreStatus.ERROR_INVALID_FIELD

        if status == DataStoreStatus.SUCCESS:
            value = self.data[key][field]

        return status, value
    
    def erase(self, key, field) -> DataStoreStatus:
        status = DataStoreStatus.SUCCESS
        
        if key not in self.data:
            status = DataStoreStatus.ERROR_INVALID_KEY
        
        if status == DataStoreStatus.SUCCESS and field not in self.data[key]:
            status = DataStoreStatus.ERROR_INVALID_FIELD

        if status == DataStoreStatus.SUCCESS:
            self.data[key].pop(field)

        return status
    
    def remove(self, key):
        status = DataStoreStatus.SUCCESS
        
        if key not in self.data:
            status = DataStoreStatus.ERROR_INVALID_KEY

        if status == DataStoreStatus.SUCCESS:
            self.data.pop(key)

        return status

    def __len__(self):
        return len(self.data)
    