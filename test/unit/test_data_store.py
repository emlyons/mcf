import unittest

from mcf.data_store import DataStore
from mcf.data_store import DataStoreStatus

class TestDataStore(unittest.TestCase):
	def setUp(self):
		self.data_store = DataStore()

	def tearDown(self):
		return

	def test_init(self):
		self.assertIsNotNone(self.data_store)

	def test_put(self):
		key = 'key'
		field = 'field'
		value = 123

		status = self.data_store.put(key, field, value)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_put_collision(self):
		key = 'key'
		field = 'field'
		value1 = 123
		value2 = 321

		status = self.data_store.put(key, field, value1)
		status = self.data_store.put(key, field, value2)

		self.assertEqual(DataStoreStatus.ERROR_FIELD_COLLISION, status)

	def test_put_replace(self):
		key = 'key'
		field = 'field'
		value1 = 123
		value2 = 321

		status = self.data_store.put(key, field, value1)
		self.data_store.erase(key, field)
		status = self.data_store.put(key, field, value2)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_put_replace_after_remove(self):
		key = 'key'
		field = 'field'
		value1 = 123
		value2 = 321

		status = self.data_store.put(key, field, value1)
		self.data_store.remove(key)
		status = self.data_store.put(key, field, value2)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_get(self):
		key = 'key'
		field = 'field'
		value = 123

		_ = self.data_store.put(key, field, value)
		status, observed_value = self.data_store.get(key, field)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)
		self.assertEqual(value, observed_value)

	def test_get_ERROR_INVALID_KEY(self):
		key = 'key'
		field = 'field'

		status, observed_value = self.data_store.get(key, field)
		
		self.assertEqual(DataStoreStatus.ERROR_INVALID_KEY, status)

	def test_get_ERROR_INVALID_FIELD(self):
		key = 'key'
		field = 'field'
		value = 123

		_ = self.data_store.put(key, field, value)
		status, observed_value = self.data_store.get(key, 'invalid field')
		
		self.assertEqual(DataStoreStatus.ERROR_INVALID_FIELD, status)

	def test_get_collision_replace(self):
		key = 'key'
		field = 'field'
		value1 = 123
		value2 = 321

		_ = self.data_store.put(key, field, value1)
		_ = self.data_store.put(key, field, value2)
		status, observed_value = self.data_store.get(key, field)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)
		self.assertEqual(value1, observed_value)

	def test_get_replaced(self):
		key = 'key'
		field = 'field'
		value1 = 123
		value2 = 321

		_ = self.data_store.put(key, field, value1)
		self.data_store.erase(key, field)
		_ = self.data_store.put(key, field, value2)
		status, observed_value = self.data_store.get(key, field)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)
		self.assertEqual(value2, observed_value)

	def test_get_replaced_after_remove(self):
		key = 'key'
		field = 'field'
		value1 = 123
		value2 = 321

		_ = self.data_store.put(key, field, value1)
		self.data_store.remove(key)
		_ = self.data_store.put(key, field, value2)
		status, observed_value = self.data_store.get(key, field)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)
		self.assertEqual(value2, observed_value)

	def test_erase(self):
		key = 'key'
		field = 'field'
		value = 123

		_ = self.data_store.put(key, field, value)
		status = self.data_store.erase(key, field)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_erase_ERROR_INVALID_KEY(self):
		key = 'key'
		field = 'field'

		status = self.data_store.erase(key, field)
		
		self.assertEqual(DataStoreStatus.ERROR_INVALID_KEY, status)

	def test_erase_ERROR_INVALID_FIELD(self):
		key = 'key'
		field = 'field'
		value = 123

		_ = self.data_store.put(key, field, value)
		status = self.data_store.erase(key, 'invalid field')
		
		self.assertEqual(DataStoreStatus.ERROR_INVALID_FIELD, status)

	def test_remove(self):
		key = 'key'
		field = 'field'
		value = 123

		_ = self.data_store.put(key, field, value)
		status = self.data_store.remove(key)
		
		self.assertEqual(DataStoreStatus.SUCCESS, status)

	def test_remove_ERROR_INVALID_KEY(self):
		key = 'key'

		status = self.data_store.remove(key)
		
		self.assertEqual(DataStoreStatus.ERROR_INVALID_KEY, status)
	

if __name__ == '__main__':
	unittest.main()
