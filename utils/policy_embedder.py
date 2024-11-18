import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import logging
from typing import List, Dict

class PolicyEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.vector_store_path = os.path.join("vector_stores", "policy_vectors")
        
        # Ensure vector store directory exists
        os.makedirs(self.vector_store_path, exist_ok=True)
        
    def _prepare_policy_text(self, policy: Dict) -> List[str]:
        """Convert policy dictionary to searchable text chunks"""
        policy_texts = []
        
        # Convert nested policy details to searchable text
        def dict_to_text(d: Dict, prefix: str = "") -> List[str]:
            texts = []
            for k, v in d.items():
                if isinstance(v, dict):
                    texts.extend(dict_to_text(v, f"{prefix}{k} - "))
                elif isinstance(v, list):
                    texts.append(f"{prefix}{k}: {', '.join(map(str, v))}")
                else:
                    texts.append(f"{prefix}{k}: {v}")
            return texts
        
        # Create structured text from policy
        policy_text = "\n".join(dict_to_text(policy))
        
        # Split into chunks
        return self.text_splitter.split_text(policy_text)
        
    def create_vector_store(self, policies_data: Dict[str, Dict]):
        """Create vector store from policies"""
        try:
            all_texts = []
            metadatas = []
            
            for policy_number, policy in policies_data.items():
                policy_chunks = self._prepare_policy_text(policy)
                all_texts.extend(policy_chunks)
                metadatas.extend([{"policy_number": policy_number}] * len(policy_chunks))
            
            self.vector_store = FAISS.from_texts(
                texts=all_texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            self.vector_store.save_local(self.vector_store_path)
            logging.info(f"Created vector store with {len(all_texts)} policy chunks")
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            raise
        
    def load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings
                )
                logging.info("Loaded existing vector store")
                return True
            return False
        except Exception as e:
            logging.error(f"Error loading vector store: {str(e)}")
            return False
    
    def search_policies(self, query: str, k: int = 3) -> List[Dict]:
        """Search policies using semantic similarity"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
                
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, doc.metadata, score) for doc, score in results]
        except Exception as e:
            logging.error(f"Error searching policies: {str(e)}")
            return []