import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.services.search_service import SearchService

class TestSearchService(unittest.TestCase):
    @patch('app.services.search_service.SentenceTransformer')
    @patch('app.services.search_service.faiss')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='[{"id": "1", "text": "test"}]')
    @patch('os.path.exists')
    def test_search_service_initialization(self, mock_exists, mock_open, mock_faiss, mock_st):
        mock_exists.return_value = True
        
        # Mock index
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        
        service = SearchService()
        
        # Reset singleton for clean slate if needed
        SearchService._instance = None
        
        self.assertIsNotNone(service.model)
        self.assertIsNotNone(service.index)
        self.assertEqual(len(service.documents_list), 1)

    @patch('app.services.search_service.SentenceTransformer')
    @patch('app.services.search_service.faiss')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='[{"id": "1", "text": "test doc"}]')
    @patch('os.path.exists')
    def test_search_documents(self, mock_exists, mock_open, mock_faiss, mock_st):
        mock_exists.return_value = True
        
        # Setup mocks
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        # Mock search return: distances, indices
        mock_index.search.return_value = (np.array([[0.5]]), np.array([[0]]))
        
        # Init service
        SearchService._instance = None # Reset singleton
        service = SearchService()
        
        results = service.search_documents("query", top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], "1")
        self.assertEqual(results[0]['score'], 0.5)

if __name__ == '__main__':
    unittest.main()
