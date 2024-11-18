from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional
import json
import logging
from utils.data_loader import DataLoader
from utils.policy_embedder import PolicyEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyValidationAgent:
    def __init__(self, groq_api_key: str, model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.2,
            max_tokens=2048
        )
        self.data_loader = DataLoader()
        self.policy_embedder = PolicyEmbedder()
        
        # Initialize embeddings and load policies
        self._initialize_data()

    def _initialize_data(self):
        """Initialize policy data and embeddings"""
        try:
            # Load policies from data loader
            self.policies_data = self.data_loader.load_all_policies()
            
            # Initialize embeddings if we have policy data
            if self.policies_data:
                if not self.policy_embedder.load_vector_store():
                    self.policy_embedder.create_vector_store(self.policies_data)
        except Exception as e:
            logger.error(f"Error initializing policy data: {str(e)}")
            self.policies_data = {}
        
    def _load_policies(self) -> Dict:
        """Load policy documents from text files"""
        policies = {}
        try:
            policies_dir = "policies_data"
            if not os.path.exists(policies_dir):
                os.makedirs(policies_dir)
                
            if os.path.exists(f"{policies_dir}/policies.txt"):
                with open(f"{policies_dir}/policies.txt", 'r') as f:
                    policies = json.loads(f.read())
                logging.info(f"Loaded {len(policies)} policies")
            else:
                logging.warning("No policies file found")
        except Exception as e:
            logging.error(f"Error loading policies: {str(e)}")
        return policies
    
    def _get_response(self, messages: List[dict]) -> str:
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return f"Error: {str(e)}"
    
    def validate_policy(self, policy_number: str, claim_details: Dict) -> Dict:
        """Validate if a claim is covered under the policy"""
        try:
            policy_data = self.policies_data.get(policy_number)
            
            if not policy_data:
                return {
                    "valid": False,
                    "error": "Policy number not found",
                    "details": None
                }
            
            # Get relevant policy sections using embeddings
            policy_sections = self.policy_embedder.search_policies(
                f"coverage details for {claim_details.get('claim_type')}"
            )
            
            messages = [
                SystemMessage(content=f"""You are a Policy Validation Oracle specialized in insurance policy verification.
                
                Policy Details:
                {json.dumps(policy_data, indent=2)}
                
                Relevant Policy Sections:
                {json.dumps(policy_sections, indent=2)}
                
                Claim Details:
                {json.dumps(claim_details, indent=2)}
                
                Analyze the claim against the policy terms and provide a structured response including:
                1. Coverage Verification
                2. Policy Limits Check
                3. Exclusions Analysis
                4. Terms Compliance
                5. Documentation Requirements"""),
                HumanMessage(content="Validate this claim against the policy terms.")
            ]
            
            validation_result = self._get_response(messages)
            
            return {
                "valid": True,
                "policy_data": policy_data,
                "validation_details": validation_result
            }
            
        except Exception as e:
            logger.error(f"Error in policy validation: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "details": None
            }
    
    def check_policy_limits(self, policy_number: str, claim_amount: float) -> Dict:
        """Check if claim amount is within policy limits"""
        policy_data = self.policies_data.get(policy_number)
        
        if not policy_data:
            return {"valid": False, "error": "Policy not found"}
            
        coverage_limit = policy_data.get('coverage_limit', 0)
        remaining_coverage = policy_data.get('remaining_coverage', coverage_limit)
        
        return {
            "valid": claim_amount <= remaining_coverage,
            "coverage_limit": coverage_limit,
            "remaining_coverage": remaining_coverage,
            "claim_amount": claim_amount,
            "within_limit": claim_amount <= remaining_coverage
        }
    
    def verify_documentation(self, policy_number: str, claim_type: str, provided_docs: List[str]) -> Dict:
        """Verify documentation requirements using semantic search"""
        try:
            # Search for documentation requirements
            doc_requirements = self.policy_embedder.search_policies(
                f"documentation requirements for {claim_type} claims in policy {policy_number}"
            )
            
            messages = [
                SystemMessage(content=f"""Verify documentation completeness.
                
                Policy Requirements:
                {json.dumps(doc_requirements, indent=2)}
                
                Provided Documents:
                {json.dumps(provided_docs, indent=2)}
                
                Analyze:
                1. Required vs Provided Documents
                2. Missing Documents
                3. Additional Requirements"""),
                HumanMessage(content="Verify documentation completeness")
            ]
            
            verification_result = self._get_response(messages)
            
            return {
                "verification_result": verification_result,
                "provided_documents": provided_docs,
                "policy_requirements": doc_requirements
            }
            
        except Exception as e:
            logging.error(f"Error in documentation verification: {str(e)}")
            return {
                "verification_result": f"Error: {str(e)}",
                "provided_documents": provided_docs,
                "policy_requirements": None
            }
    
    def get_policy_summary(self, policy_number: str) -> str:
        """Get a human-readable summary of policy terms"""
        try:
            # First try to get policy from loaded data
            policy_data = self.policies_data.get(policy_number)
            
            if not policy_data:
                return f"Policy {policy_number} not found"
            
            messages = [
                SystemMessage(content=f"""Create a clear, concise summary of this insurance policy.
                
                Policy Details:
                {json.dumps(policy_data, indent=2)}
                
                Include:
                1. Coverage Overview
                2. Key Terms and Conditions
                3. Important Exclusions
                4. Claim Requirements
                5. Coverage Limits"""),
                HumanMessage(content="Generate a policy summary.")
            ]
            
            return self._get_response(messages)
            
        except Exception as e:
            logger.error(f"Error generating policy summary: {str(e)}")
            return f"Error summarizing policy: {str(e)}"
