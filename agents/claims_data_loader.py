import json
import os
from typing import List, Dict
import logging

class ClaimsDataLoader:
    def __init__(self, data_directory: str = "claims_data"):
        self.data_directory = data_directory
        self._ensure_data_directory()
        
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            
    def load_claims_history(self) -> List[Dict]:
        """Load historical claims data from text files"""
        claims_data = []
        try:
            # Load the main claims history file
            history_file = os.path.join(self.data_directory, "claims_history.txt")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    claims_data = json.loads(f.read())
                logging.info(f"Loaded {len(claims_data)} historical claims")
            else:
                logging.warning("No claims history file found")
        except Exception as e:
            logging.error(f"Error loading claims history: {str(e)}")
            claims_data = []
            
        return claims_data
    
    def save_new_claim(self, claim_data: Dict):
        """Save a new claim to the history"""
        try:
            claims_data = self.load_claims_history()
            claims_data.append(claim_data)
            
            history_file = os.path.join(self.data_directory, "claims_history.txt")
            with open(history_file, 'w') as f:
                json.dump(claims_data, f, indent=4)
                
            logging.info(f"Saved new claim {claim_data.get('claim_id')}")
        except Exception as e:
            logging.error(f"Error saving claim: {str(e)}")
    
    def filter_claims(self, 
                     claim_type: str = None, 
                     min_amount: float = None,
                     max_amount: float = None,
                     status: str = None) -> List[Dict]:
        """Filter claims based on criteria"""
        claims_data = self.load_claims_history()
        filtered_claims = claims_data
        
        if claim_type:
            filtered_claims = [c for c in filtered_claims if c['claim_type'] == claim_type]
        
        if min_amount is not None:
            filtered_claims = [c for c in filtered_claims if c['amount'] >= min_amount]
            
        if max_amount is not None:
            filtered_claims = [c for c in filtered_claims if c['amount'] <= max_amount]
            
        if status:
            filtered_claims = [c for c in filtered_claims if c['status'] == status]
            
        return filtered_claims
    
    def get_claim_statistics(self) -> Dict:
        """Calculate statistics from claims history"""
        claims_data = self.load_claims_history()
        
        if not claims_data:
            return {}
        
        total_claims = len(claims_data)
        approved_claims = len([c for c in claims_data if c['status'] == 'Approved'])
        total_amount = sum(c['amount'] for c in claims_data)
        avg_processing_time = sum(c['processing_time'] for c in claims_data) / total_claims
        
        return {
            'total_claims': total_claims,
            'approved_claims': approved_claims,
            'approval_rate': (approved_claims / total_claims) * 100 if total_claims > 0 else 0,
            'total_amount': total_amount,
            'average_amount': total_amount / total_claims if total_claims > 0 else 0,
            'average_processing_time': avg_processing_time,
            'claim_types': {
                claim_type: len([c for c in claims_data if c['claim_type'] == claim_type])
                for claim_type in set(c['claim_type'] for c in claims_data)
            }
        }