from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional, Any, Union
import json
import logging
from datetime import datetime
from utils.data_loader import DataLoader
from utils.claims_embedder import ClaimsEmbedder
import numpy as np

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimsAnalysisAgent:
    def __init__(self, groq_api_key: str, model: str = "mixtral-8x7b-32768"):
        logger.info("Initializing ClaimsAnalysisAgent")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=2048
        )
        self.data_loader = DataLoader()
        self.claims_embedder = ClaimsEmbedder()
        self.policy_context = None
        
        # Initialize embeddings with historical data
        self._initialize_embeddings()

    def _ensure_serializable(self, data: Any) -> Union[Dict, List, str, float, int, bool, None]:
        """Convert any non-serializable types to standard Python types"""
        if isinstance(data, dict):
            return {key: self._ensure_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._ensure_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)  # Convert any other types to string
        
    def _initialize_embeddings(self):
        """Initialize or load vector stores"""
        try:
            logger.info("Loading claims history for embeddings")
            claims_data = self.data_loader.load_claims_history()
            if claims_data:
                logger.info(f"Found {len(claims_data)} claims in history")
                if not self.claims_embedder.load_vector_store():
                    logger.info("Creating new vector store for claims")
                    self.claims_embedder.create_vector_store(claims_data)
            else:
                logger.warning("No claims data found for embeddings")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")

    def set_policy_context(self, policy_data: Dict):
        """Set the insurance policy context for analysis"""
        try:
            logger.info(f"Setting policy context for policy number: {policy_data.get('policy_number', 'unknown')}")
            self.policy_context = policy_data
        except Exception as e:
            logger.error(f"Error setting policy context: {str(e)}")

    def _get_response(self, messages: List[dict]) -> str:
        """Get response from LLM"""
        try:
            if self.policy_context:
                logger.debug("Adding policy context to messages")
                context_message = SystemMessage(content=f"""You are the Claims Analysis Oracle, a specialized AI system for analyzing insurance claims.
                
                Current Policy Information:
                Policy Type: {self.policy_context.get('policy_type', 'N/A')}
                Coverage Limits: ${self.policy_context.get('coverage_limit', 0):,.2f}
                Deductible: ${self.policy_context.get('deductible', 0):,.2f}
                Policy Status: {self.policy_context.get('status', 'Unknown')}
                
                Key Guidelines:
                1. Analyze claim validity against policy terms
                2. Check for coverage limits and exclusions
                3. Identify potential fraud indicators
                4. Consider claim history patterns
                5. Recommend optimal settlement approaches
                """)
                messages.insert(0, context_message)
            
            logger.debug("Sending request to LLM")
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def analyze_claim(self, claim_details: Dict) -> str:
        """Analyze a new insurance claim and provide recommendations"""
        try:
            logger.info(f"Analyzing claim {claim_details.get('claim_id', 'unknown')}")
            
            # Get similar claims using semantic search
            similar_claims = self.claims_embedder.find_similar_claims(claim_details)
            logger.info(f"Found {len(similar_claims)} similar claims")
            
            # Convert similar claims to JSON serializable format
            serializable_similar_claims = []
            for claim in similar_claims:
                serializable_claim = {
                    'content': claim['content'],
                    'metadata': claim['metadata'],
                    'similarity_score': float(claim['similarity_score'])  # Convert float32 to float
                }
                serializable_similar_claims.append(serializable_claim)
            
            # Get policy claim history
            claim_history = self.data_loader.search_claims(
                policy_number=claim_details.get('policy_number')
            )
            logger.info(f"Found {len(claim_history)} claims in policy history")
            
            # Ensure all numeric values are standard Python types
            serializable_claim_details = {}
            for key, value in claim_details.items():
                if hasattr(value, 'dtype'):  # Check if it's a numpy type
                    serializable_claim_details[key] = float(value)
                else:
                    serializable_claim_details[key] = value
            
            # Convert claim history to serializable format
            serializable_history = []
            for hist_claim in claim_history:
                serializable_hist_claim = {}
                for key, value in hist_claim.items():
                    if hasattr(value, 'dtype'):  # Check if it's a numpy type
                        serializable_hist_claim[key] = float(value)
                    else:
                        serializable_hist_claim[key] = value
                serializable_history.append(serializable_hist_claim)
            
            messages = [
                SystemMessage(content=f"""Analyze this insurance claim thoroughly.
                
                Claim Details:
                {json.dumps(serializable_claim_details, indent=2)}
                
                Similar Claims History:
                {json.dumps(serializable_similar_claims, indent=2)}
                
                Policy Claim History:
                {json.dumps(serializable_history, indent=2)}
                
                Provide a detailed analysis in the following format:
                
                1. 📋 Claim Overview
                   - Basic details review
                   - Initial assessment
                   - Completeness check
                
                2. 🔍 Historical Analysis
                   - Similar claims patterns
                   - Policy claim history
                   - Typical outcomes
                
                3. ⚠️ Risk Assessment
                   - Policy compliance
                   - Documentation status
                   - Potential issues
                
                4. 💰 Cost Analysis
                   - Amount reasonableness
                   - Historical comparisons
                   - Cost factors
                
                5. 📊 Settlement Recommendation
                   - Suggested action
                   - Amount recommendation
                   - Justification
                
                6. 📝 Processing Notes
                   - Required actions
                   - Timeline estimate
                   - Special considerations
                """),
                HumanMessage(content="Please analyze this claim and provide recommendations")
            ]
            
            logger.debug("Requesting claim analysis from LLM")
            response = self._get_response(messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing claim: {str(e)}")
            return f"Error analyzing claim: {str(e)}"

    def get_similar_claims(self, claim_details: Dict) -> str:
        """Find and analyze similar historical claims using embeddings"""
        try:
            logger.info(f"Searching for claims similar to {claim_details.get('claim_id', 'unknown')}")
            
            # Get similar claims using semantic search
            similar_claims = self.claims_embedder.find_similar_claims(claim_details)
            
            if not similar_claims:
                logger.info("No similar claims found")
                return f"""
                🔍 No similar claims found for:
                - Policy: {claim_details.get('policy_number')}
                - Type: {claim_details.get('claim_type')}
                
                This appears to be the first claim of this type for this policy.
                """
            
            logger.info(f"Found {len(similar_claims)} similar claims")
            
            # Format claims for analysis
            claims_analysis = []
            total_amount = 0
            approval_count = 0
            
            for claim in similar_claims:
                content = claim['content']
                amount = float(content.get('Amount', '0').replace('$', '').strip())
                status = content.get('Status', 'Unknown')
                
                total_amount += amount
                if status.lower() == 'approved':
                    approval_count += 1
                
                # Convert numpy float32 to regular float for JSON serialization
                similarity_score = float(claim['similarity_score'])
                
                claims_analysis.append({
                    'claim_id': str(content.get('Claim ID')),
                    'amount': float(amount),
                    'status': str(status),
                    'similarity_score': float(similarity_score)
                })
            
            # Calculate statistics
            avg_amount = total_amount / len(similar_claims)
            approval_rate = (approval_count / len(similar_claims)) * 100
            
            logger.info(f"Calculated statistics: avg_amount=${avg_amount:.2f}, approval_rate={approval_rate:.1f}%")
            
            summary = f"""
            📊 Similar Claims Analysis
            
            Found {len(similar_claims)} similar claims:
            • Average Amount: ${avg_amount:,.2f}
            • Approval Rate: {approval_rate:.1f}%
            
            Most Similar Claims:
            """
            
            # Add details of each similar claim
            for claim in claims_analysis[:3]:
                summary += f"""
                Claim {claim['claim_id']}:
                • Amount: ${claim['amount']:,.2f}
                • Status: {claim['status']}
                • Similarity: {(1 - claim['similarity_score'])*100:.1f}%
                """
            
            # Add analysis and recommendations
            messages = [
                SystemMessage(content=f"""Analyze these similar claims and provide insights.
                
                Current Claim:
                {json.dumps(claim_details, indent=2)}
                
                Similar Claims:
                {json.dumps(claims_analysis, indent=2)}
                
                Provide specific insights about:
                1. Amount patterns
                2. Approval likelihood
                3. Processing considerations
                4. Risk factors
                """),
                HumanMessage(content="Analyze similar claims patterns")
            ]
            
            logger.debug("Requesting similar claims analysis from LLM")
            analysis = self._get_response(messages)
            
            return f"{summary}\n\n💡 Analysis:\n{analysis}"
            
        except Exception as e:
            logger.error(f"Error in get_similar_claims: {str(e)}")
            return f"Error finding similar claims: {str(e)}"

    def detect_fraud_indicators(self, claim_details: Dict) -> List[str]:
        """Detect potential fraud indicators in a claim"""
        try:
            logger.info(f"Checking fraud indicators for claim {claim_details.get('claim_id', 'unknown')}")
            red_flags = []
            
            # Get recent claims for this policy
            recent_claims = self.data_loader.search_claims(
                policy_number=claim_details.get('policy_number'),
                date_from=(datetime.strptime(claim_details.get('date_filed'), '%Y-%m-%d')
                          .replace(month=1).strftime('%Y-%m-%d'))
            )
            
            logger.info(f"Found {len(recent_claims)} recent claims for policy")
            
            # Check for multiple claims in short period
            if len(recent_claims) > 5:
                red_flags.append("High frequency of claims")
            
            # Get similar claims for amount comparison
            similar_claims = self.claims_embedder.find_similar_claims(
                claim_details,
                k=10
            )
            
            if similar_claims:
                # Calculate average amount from similar claims
                amounts = [float(claim['content'].get('Amount', '0').replace('$', '').strip()) 
                          for claim in similar_claims]
                avg_amount = sum(amounts) / len(amounts)
                
                if claim_details.get('amount', 0) > avg_amount * 2:
                    red_flags.append(f"Amount (${claim_details.get('amount', 0):,.2f}) significantly higher than average (${avg_amount:,.2f})")
            
            # Document completeness check
            required_docs = self.get_required_documents(claim_details.get('claim_type'))
            provided_docs = claim_details.get('documents_provided', [])
            missing_docs = set(required_docs) - set(provided_docs)
            
            if missing_docs:
                red_flags.append(f"Missing required documents: {', '.join(missing_docs)}")
            
            logger.info(f"Identified {len(red_flags)} potential fraud indicators")
            return red_flags
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {str(e)}")
            return ["Error in fraud detection analysis"]

    def get_required_documents(self, claim_type: str) -> List[str]:
        """Get list of required documents for a claim type"""
        logger.debug(f"Getting required documents for claim type: {claim_type}")
        document_requirements = {
            "Emergency Care": [
                "Hospital Report",
                "Medical Bills",
                "Treatment Records",
                "Emergency Room Documentation"
            ],
            "Prescription": [
                "Prescription",
                "Pharmacy Bill",
                "Doctor's Note"
            ],
            "Specialist Visit": [
                "Referral Letter",
                "Specialist Report",
                "Medical Bills",
                "Treatment Plan"
            ],
            "Preventive Care": [
                "Provider Report",
                "Medical Bills",
                "Preventive Care Schedule"
            ]
        }
        
        return document_requirements.get(claim_type, ["Medical Documentation", "Bills"])
    
    def suggest_settlement_amount(self, claim_details: Dict) -> tuple[float, str]:
        """Suggest optimal settlement amount based on similar claims and policy terms"""
        try:
            logger.info(f"Calculating suggested settlement for claim {claim_details.get('claim_id', 'unknown')}")
            
            # Get similar claims for amount comparison
            similar_claims = self.claims_embedder.find_similar_claims(
                claim_details,
                k=5
            )
            
            claim_amount = float(claim_details.get('amount', 0))
            
            if not similar_claims:
                logger.info("No similar claims found, using default settlement ratio")
                # Default to 80% of claim amount if no similar claims
                return claim_amount * 0.8, """
                No similar claims found for comparison.
                Using standard settlement ratio of 80%.
                """
            
            # Calculate average settlement ratio from similar claims
            settlement_ratios = []
            total_settlement = 0
            total_original = 0
            
            for claim in similar_claims:
                content = claim['content']
                original_amount = float(content.get('Amount', '0').replace('$', '').strip())
                settlement_amount = float(content.get('Settlement Amount', '0').replace('$', '').strip() or '0')
                
                if original_amount > 0 and settlement_amount > 0:
                    ratio = settlement_amount / original_amount
                    settlement_ratios.append(ratio)
                    total_settlement += settlement_amount
                    total_original += original_amount
            
            if not settlement_ratios:
                logger.warning("No valid settlement ratios found")
                return claim_amount * 0.8, "No valid settlement data found. Using standard ratio."
            
            # Calculate average ratio and overall ratio
            avg_ratio = sum(settlement_ratios) / len(settlement_ratios)
            overall_ratio = total_settlement / total_original
            
            # Calculate suggested amount
            suggested_amount = claim_amount * avg_ratio
            
            # Prepare explanation
            explanation = f"""
            💡 Settlement Analysis:
            • Based on {len(similar_claims)} similar claims
            • Average settlement ratio: {avg_ratio:.1%}
            • Overall settlement ratio: {overall_ratio:.1%}
            • Original claim amount: ${claim_amount:,.2f}
            • Suggested settlement: ${suggested_amount:,.2f}
            
            Similar Claims Settlement Pattern:
            """
            
            # Add details of similar claims settlements
            for i, claim in enumerate(similar_claims[:3], 1):
                content = claim['content']
                original = float(content.get('Amount', '0').replace('$', '').strip())
                settlement = float(content.get('Settlement Amount', '0').replace('$', '').strip() or '0')
                if original > 0 and settlement > 0:
                    ratio = settlement / original
                    explanation += f"""
                    Claim {i}:
                    • Original: ${original:,.2f}
                    • Settlement: ${settlement:,.2f}
                    • Ratio: {ratio:.1%}
                    """
            
            messages = [
                SystemMessage(content=f"""Analyze this settlement recommendation:
                
                Current Claim:
                {json.dumps(claim_details, indent=2)}
                
                Settlement Analysis:
                - Suggested Amount: ${suggested_amount:,.2f}
                - Average Ratio: {avg_ratio:.1%}
                - Similar Claims: {len(similar_claims)}
                
                Provide a brief justification for this settlement amount."""),
                HumanMessage(content="Provide settlement justification")
            ]
            
            justification = self._get_response(messages)
            explanation += f"\n\n💭 Justification:\n{justification}"
            
            logger.info(f"Calculated suggested settlement: ${suggested_amount:,.2f}")
            return suggested_amount, explanation
            
        except Exception as e:
            logger.error(f"Error calculating settlement amount: {str(e)}")
            # Return default values in case of error
            return claim_details.get('amount', 0) * 0.8, f"Error calculating settlement: {str(e)}"
    
    def get_settlement_metrics(self, claim_details: Dict) -> Dict:
        """Get additional settlement metrics and insights"""
        try:
            logger.info(f"Calculating settlement metrics for claim {claim_details.get('claim_id', 'unknown')}")
            
            # Get similar claims
            similar_claims = self.claims_embedder.find_similar_claims(
                claim_details,
                k=10
            )
            
            if not similar_claims:
                return {
                    "average_processing_time": None,
                    "approval_rate": None,
                    "settlement_range": None,
                    "confidence_score": None
                }
            
            # Calculate metrics
            processing_times = []
            settlement_amounts = []
            approved_count = 0
            
            for claim in similar_claims:
                content = claim['content']
                
                # Processing time
                if 'Processing Time' in content:
                    try:
                        processing_time = float(content['Processing Time'])
                        processing_times.append(processing_time)
                    except ValueError:
                        pass
                
                # Settlement amounts
                amount_str = content.get('Settlement Amount', '0').replace('$', '').strip()
                try:
                    settlement_amount = float(amount_str)
                    if settlement_amount > 0:
                        settlement_amounts.append(settlement_amount)
                except ValueError:
                    pass
                
                # Approval count
                if content.get('Status', '').lower() == 'approved':
                    approved_count += 1
            
            # Calculate final metrics
            metrics = {
                "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else None,
                "approval_rate": (approved_count / len(similar_claims)) * 100 if similar_claims else None,
                "settlement_range": {
                    "min": min(settlement_amounts) if settlement_amounts else None,
                    "max": max(settlement_amounts) if settlement_amounts else None,
                    "avg": sum(settlement_amounts) / len(settlement_amounts) if settlement_amounts else None
                },
                "confidence_score": len(similar_claims) / 10  # Scale of 0-1 based on number of similar claims
            }
            
            logger.info("Successfully calculated settlement metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating settlement metrics: {str(e)}")
            return {
                "error": str(e),
                "average_processing_time": None,
                "approval_rate": None,
                "settlement_range": None,
                "confidence_score": None
            }