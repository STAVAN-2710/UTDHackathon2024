import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimsEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.base_path = "vector_stores"
        self.vector_store_path = os.path.join(self.base_path, "claims_vectors")
        
        # Ensure vector store directory exists
        os.makedirs(self.base_path, exist_ok=True)
        
    def _prepare_claim_text(self, claim: Dict) -> str:
        """Convert claim to searchable text"""
        try:
            return f"""
            Claim ID: {claim.get('claim_id', 'N/A')}
            Policy: {claim.get('policy_number', 'N/A')}
            Type: {claim.get('claim_type', 'N/A')}
            Amount: ${claim.get('amount', 0):.2f}
            Description: {claim.get('description', 'N/A')}
            Status: {claim.get('status', 'Pending')}
            Documents: {', '.join(claim.get('documents_provided', []))}
            Date Filed: {claim.get('date_filed', 'N/A')}
            """
        except Exception as e:
            logger.error(f"Error preparing claim text: {str(e)}")
            return ""
        
    def create_vector_store(self, claims_data: List[Dict]):
        """Create vector store from claims"""
        try:
            logger.info(f"Creating vector store from {len(claims_data)} claims")
            
            # Filter out any claims with invalid data
            valid_claims = [claim for claim in claims_data if claim.get('claim_id')]
            
            texts = []
            metadatas = []
            
            for claim in valid_claims:
                claim_text = self._prepare_claim_text(claim)
                if claim_text.strip():  # Only add non-empty texts
                    texts.append(claim_text)
                    metadatas.append({
                        "claim_id": claim["claim_id"],
                        "policy_number": claim.get("policy_number", "N/A"),
                        "claim_type": claim.get("claim_type", "N/A")
                    })
            
            if not texts:
                logger.warning("No valid claims to create vector store")
                return
            
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save vector store
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"Successfully created vector store with {len(texts)} claims")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
        
    def load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                logger.info("Loading existing vector store")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings
                )
                return True
            logger.info("No existing vector store found")
            return False
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def find_similar_claims(self, query_claim: Dict, k: int = 5) -> List[Dict]:
        """Find similar claims using semantic search"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            query_text = self._prepare_claim_text(query_claim)
            if not query_text.strip():
                logger.warning("Invalid query claim")
                return []
            
            results = self.vector_store.similarity_search_with_score(query_text, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # Parse the document content
                content_lines = doc.page_content.strip().split('\n')
                claim_data = {}
                for line in content_lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        claim_data[key.strip()] = value.strip()
                
                formatted_results.append({
                    'content': claim_data,
                    'metadata': doc.metadata,
                    'similarity_score': score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error finding similar claims: {str(e)}")
            return []