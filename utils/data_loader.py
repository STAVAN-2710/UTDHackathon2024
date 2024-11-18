import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.claims_dir = os.path.join(data_dir, "claims_data")
        self.policies_dir = os.path.join(data_dir, "policies_data")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.claims_dir, exist_ok=True)
        os.makedirs(self.policies_dir, exist_ok=True)
    
    def load_claims_history(self) -> List[Dict]:
        """Load claims history from file"""
        try:
            claims_file = os.path.join(self.claims_dir, "claims_history.txt")
            if os.path.exists(claims_file):
                with open(claims_file, 'r') as f:
                    claims_data = json.loads(f.read())
                logger.info(f"Loaded {len(claims_data)} claims from history")
                return claims_data
            else:
                logger.warning("No claims history file found")
                return []
        except Exception as e:
            logger.error(f"Error loading claims history: {str(e)}")
            return []
    
    def save_claim(self, claim_data: Dict) -> bool:
        """Save a new claim to history"""
        try:
            claims_data = self.load_claims_history()
            claims_data.append(claim_data)
            
            claims_file = os.path.join(self.claims_dir, "claims_history.txt")
            with open(claims_file, 'w') as f:
                json.dump(claims_data, f, indent=4)
            
            logger.info(f"Saved claim {claim_data.get('claim_id')} to history")
            return True
        except Exception as e:
            logger.error(f"Error saving claim: {str(e)}")
            return False
    
    def load_policy(self, policy_number: str) -> Optional[Dict]:
        """Load specific policy data"""
        try:
            policies_file = os.path.join(self.policies_dir, "policies.txt")
            if os.path.exists(policies_file):
                with open(policies_file, 'r') as f:
                    policies_data = json.loads(f.read())
                return policies_data.get(policy_number)
            else:
                logger.warning("No policies file found")
                return None
        except Exception as e:
            logger.error(f"Error loading policy: {str(e)}")
            return None
    
    def load_all_policies(self) -> Dict:
        """Load all policies"""
        try:
            policies_file = os.path.join(self.policies_dir, "policies.txt")
            if os.path.exists(policies_file):
                with open(policies_file, 'r') as f:
                    return json.loads(f.read())
            else:
                logger.warning("No policies file found")
                return {}
        except Exception as e:
            logger.error(f"Error loading policies: {str(e)}")
            return {}
    
    def update_policy(self, policy_number: str, updates: Dict) -> bool:
        """Update policy data"""
        try:
            policies_data = self.load_all_policies()
            if policy_number in policies_data:
                policies_data[policy_number].update(updates)
                
                policies_file = os.path.join(self.policies_dir, "policies.txt")
                with open(policies_file, 'w') as f:
                    json.dump(policies_data, f, indent=4)
                
                logger.info(f"Updated policy {policy_number}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating policy: {str(e)}")
            return False
    
    def get_claim_statistics(self, policy_number: Optional[str] = None) -> Dict:
        """Get statistics about claims"""
        claims_data = self.load_claims_history()
        
        if policy_number:
            claims_data = [c for c in claims_data if c['policy_number'] == policy_number]
        
        if not claims_data:
            return {}
        
        total_claims = len(claims_data)
        approved_claims = len([c for c in claims_data if c['status'] == 'Approved'])
        total_amount = sum(c['amount'] for c in claims_data)
        settled_amount = sum(c['settlement_amount'] for c in claims_data)
        
        return {
            'total_claims': total_claims,
            'approved_claims': approved_claims,
            'approval_rate': (approved_claims / total_claims * 100) if total_claims > 0 else 0,
            'total_amount': total_amount,
            'settled_amount': settled_amount,
            'average_amount': total_amount / total_claims if total_claims > 0 else 0,
            'average_processing_time': sum(c['processing_time'] for c in claims_data) / total_claims if total_claims > 0 else 0,
            'claim_types': {
                claim_type: len([c for c in claims_data if c['claim_type'] == claim_type])
                for claim_type in set(c['claim_type'] for c in claims_data)
            }
        }
    
    def search_claims(self, 
                     policy_number: Optional[str] = None,
                     claim_type: Optional[str] = None,
                     min_amount: Optional[float] = None,
                     max_amount: Optional[float] = None,
                     status: Optional[str] = None,
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None) -> List[Dict]:
        """Search claims with filters"""
        claims_data = self.load_claims_history()
        filtered_claims = claims_data
        
        if policy_number:
            filtered_claims = [c for c in filtered_claims if c['policy_number'] == policy_number]
        
        if claim_type:
            filtered_claims = [c for c in filtered_claims if c['claim_type'] == claim_type]
        
        if min_amount is not None:
            filtered_claims = [c for c in filtered_claims if c['amount'] >= min_amount]
        
        if max_amount is not None:
            filtered_claims = [c for c in filtered_claims if c['amount'] <= max_amount]
        
        if status:
            filtered_claims = [c for c in filtered_claims if c['status'] == status]
        
        if date_from:
            filtered_claims = [c for c in filtered_claims if c['date_filed'] >= date_from]
        
        if date_to:
            filtered_claims = [c for c in filtered_claims if c['date_filed'] <= date_to]
        
        return filtered_claims