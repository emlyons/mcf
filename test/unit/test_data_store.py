import unittest
from dataclasses import dataclass
from mcf.data_store import DataStore
from mcf.data_store import DataStoreStatus

@dataclass
class TestDataObj:
	field1: str
	field2: int

class TestDataStore(unittest.TestCase):
	def setUp(self):
		self.data_store = DataStore()

	def tearDown(self):
		return

	def test_init(self):
		self.assertIsNotNone(self.data_store)

	def test_put(self):
		key = 'key'
		value = TestDataObj("field1", 123)

		status = self.data_store.put(key, value)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_put_collision(self):
		key = 'key'
		value1 = TestDataObj("field1", 123)
		value2 = TestDataObj("field1", 123)

		_ = self.data_store.put(key, value1)
		status = self.data_store.put(key, value2)
		
		self.assertEqual(DataStoreStatus.ERROR_KEY_COLLISION, status)

	def test_put_replace(self):
		key = 'key'
		value1 = TestDataObj("field1", 123)
		value2 = TestDataObj("field1", 123)

		_ = self.data_store.put(key, value1)
		self.data_store.remove('key')
		status = self.data_store.put(key, value2)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_get(self):
		key = 'key'
		value = TestDataObj("field1", 123)

		_ = self.data_store.put(key, value)
		status, observed = self.data_store.get(key)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)
		self.assertEqual(value, observed)

	def test_get_invalid(self):
		key = 'key'

		status, _ = self.data_store.get(key)
		
		self.assertEqual(DataStoreStatus.ERROR_INVALID_KEY, status)

	def test_remove(self):
		key = 'key'
		value = TestDataObj("field1", 123)

		status = self.data_store.put(key, value)
		status = self.data_store.remove(key)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

		status, _ = self.data_store.get(key)

		self.assertEqual(DataStoreStatus.ERROR_INVALID_KEY, status)

	def test_mutate(self):
		key = 'key'
		value = TestDataObj("field1", 123)

		_ = self.data_store.put(key, value)
		_, value_ref = self.data_store.get(key)

		value_ref.field2 = 333222111

		_, observed = self.data_store.get(key)
		
		self.assertEqual(333222111, observed.field2)
	

if __name__ == '__main__':
	unittest.main()
