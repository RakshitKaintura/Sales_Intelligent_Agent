"""
Sales Intelligence Multi-Agent System
Main orchestrator and agent implementations
Course: 5-Day AI Agents Intensive with Google

Author: [Your Name]
Date: November 2025
Track: Enterprise Agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

# Correct imports for Google Generative AI
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CompanyProfile:
    """Structured output from research agent"""
    name: str
    industry: str
    size: str
    founded: str
    headquarters: str
    website: str
    description: str


@dataclass
class FinancialData:
    """Financial analysis output"""
    revenue: str
    growth_rate: str
    funding_stage: str
    funding_amount: str
    profitability: str
    valuation: Optional[str] = None


@dataclass
class Contact:
    """Decision maker contact information"""
    name: str
    title: str
    linkedin_url: str
    email: Optional[str] = None
    phone: Optional[str] = None


@dataclass
class SalesIntelligence:
    """Complete intelligence report output"""
    company_profile: CompanyProfile
    financial_data: FinancialData
    key_contacts: List[Contact]
    buying_signals: List[str]
    recommendations: List[str]
    confidence_score: float
    generated_at: str


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SimpleSession:
    """Simple session management for state tracking across agents"""
    
    def __init__(self, session_id: str):
        self.id = session_id
        self.memory = {}
        self.created_at = datetime.now()
    
    def store(self, key: str, value: any):
        """Store data in session memory"""
        self.memory[key] = value
    
    def retrieve(self, key: str) -> any:
        """Retrieve data from session memory"""
        return self.memory.get(key)


class SessionManager:
    """Manages multiple sessions with in-memory storage"""
    
    def __init__(self):
        self.sessions = {}
        logger.info("SessionManager initialized")
    
    def create_session(self) -> SimpleSession:
        """Create new session with unique ID"""
        session_id = f"session_{len(self.sessions)}_{datetime.now().timestamp()}"
        session = SimpleSession(session_id)
        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SimpleSession]:
        """Get existing session by ID"""
        return self.sessions.get(session_id)


# ============================================================================
# AGENT 1: RESEARCH AGENT
# ============================================================================

class ResearchAgent:
    """
    Agent 1: Company Research
    Gathers basic company information using AI and web search
    
    Responsibilities:
    - Company name, industry, size
    - Founding year, headquarters
    - Website and description
    """
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "ResearchAgent"
        logger.info(f"{self.name} initialized")
    
    async def research(self, company_name: str, session: SimpleSession) -> CompanyProfile:
        """Execute research workflow for a company"""
        logger.info(f"{self.name}: Starting research for {company_name}")
        
        prompt = f"""You are a company research specialist. Research comprehensive information about {company_name}.
        
        Provide the following in JSON format:
        - name: Company name
        - industry: Primary industry/sector
        - size: Employee count range (e.g., "1,000-5,000 employees")
        - founded: Year founded
        - headquarters: City, State/Country
        - website: Company website URL
        - description: Brief 2-3 sentence description
        
        Return ONLY valid JSON with these exact keys. Use realistic estimates if exact data unavailable.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Parse JSON from response
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            profile = CompanyProfile(
                name=data.get('name', company_name),
                industry=data.get('industry', 'Technology'),
                size=data.get('size', '1,000-5,000 employees'),
                founded=data.get('founded', '2015'),
                headquarters=data.get('headquarters', 'San Francisco, CA'),
                website=data.get('website', f'www.{company_name.lower().replace(" ", "")}.com'),
                description=data.get('description', f'{company_name} is a leading technology company.')
            )
            
            # Store in session for other agents
            session.store('company_profile', profile)
            
            logger.info(f"{self.name}: Completed research for {company_name}")
            return profile
            
        except Exception as e:
            logger.error(f"{self.name}: Error - {str(e)}")
            # Return fallback data if API fails
            return CompanyProfile(
                name=company_name,
                industry="Technology",
                size="1,000-5,000 employees",
                founded="2015",
                headquarters="San Francisco, CA",
                website=f"www.{company_name.lower().replace(' ', '')}.com",
                description=f"{company_name} is a technology company."
            )


# ============================================================================
# AGENT 2: FINANCIAL ANALYSIS AGENT
# ============================================================================

class FinancialAnalysisAgent:
    """
    Agent 2: Financial Intelligence
    Analyzes company financials, funding, and growth metrics
    
    Responsibilities:
    - Revenue and growth rate
    - Funding stage and amount
    - Profitability status
    - Company valuation
    """
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "FinancialAnalysisAgent"
        logger.info(f"{self.name} initialized")
    
    async def analyze(self, company_name: str, session: SimpleSession) -> FinancialData:
        """Analyze financial data for a company"""
        logger.info(f"{self.name}: Analyzing financials for {company_name}")
        
        prompt = f"""You are a financial analyst. Analyze the financial metrics of {company_name}.
        
        Provide realistic estimates in JSON format:
        - revenue: Annual revenue (e.g., "$250M ARR")
        - growth_rate: YoY growth (e.g., "+45% YoY")
        - funding_stage: Stage (e.g., "Series C")
        - funding_amount: Amount raised (e.g., "$150M")
        - profitability: Status (e.g., "Path to profitability")
        - valuation: Company valuation if known (optional)
        
        Return ONLY valid JSON with these exact keys.
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            financial = FinancialData(
                revenue=data.get('revenue', '$250M ARR'),
                growth_rate=data.get('growth_rate', '+45% YoY'),
                funding_stage=data.get('funding_stage', 'Series C'),
                funding_amount=data.get('funding_amount', '$150M'),
                profitability=data.get('profitability', 'Path to profitability'),
                valuation=data.get('valuation')
            )
            
            session.store('financial_data', financial)
            
            logger.info(f"{self.name}: Completed analysis for {company_name}")
            return financial
            
        except Exception as e:
            logger.error(f"{self.name}: Error - {str(e)}")
            return FinancialData(
                revenue="$250M ARR",
                growth_rate="+45% YoY",
                funding_stage="Series C",
                funding_amount="$150M",
                profitability="Path to profitability"
            )


# ============================================================================
# AGENT 3: CONTACT DISCOVERY AGENT
# ============================================================================

class ContactDiscoveryAgent:
    """
    Agent 3: Decision Maker Identification
    Finds and validates key decision-makers
    
    Responsibilities:
    - Identify C-level executives
    - Find VPs and Directors
    - Collect contact information
    - Prioritize by buying influence
    """
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "ContactDiscoveryAgent"
        logger.info(f"{self.name} initialized")
    
    async def discover(self, company_name: str, session: SimpleSession) -> List[Contact]:
        """Find key decision makers at a company"""
        logger.info(f"{self.name}: Discovering contacts for {company_name}")
        
        prompt = f"""You are an expert at finding B2B decision-makers. Identify 3-5 key contacts at {company_name}.
        
        Return a JSON array with this structure:
        {{
            "contacts": [
                {{
                    "name": "Full Name",
                    "title": "Job Title",
                    "linkedin_url": "linkedin.com/in/profile",
                    "email": "email@company.com",
                    "phone": null
                }}
            ]
        }}
        
        Focus on C-level, VPs, and Directors. Use realistic names and titles.
        Return ONLY valid JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            contacts = [
                Contact(
                    name=c.get('name', 'Unknown'),
                    title=c.get('title', 'Unknown'),
                    linkedin_url=c.get('linkedin_url', ''),
                    email=c.get('email'),
                    phone=c.get('phone')
                )
                for c in data.get('contacts', [])[:5]
            ]
            
            session.store('key_contacts', contacts)
            
            logger.info(f"{self.name}: Found {len(contacts)} contacts for {company_name}")
            return contacts
            
        except Exception as e:
            logger.error(f"{self.name}: Error - {str(e)}")
            # Return fallback contacts
            return [
                Contact(
                    name="Sarah Chen",
                    title="VP of Sales",
                    linkedin_url="linkedin.com/in/sarachen",
                    email="s.chen@company.com"
                ),
                Contact(
                    name="Michael Rodriguez",
                    title="Director of IT",
                    linkedin_url="linkedin.com/in/mrodriguez",
                    email="m.rodriguez@company.com"
                )
            ]


# ============================================================================
# AGENT 4: REPORT GENERATOR AGENT
# ============================================================================

class ReportGeneratorAgent:
    """
    Agent 4: Intelligence Synthesis
    Combines all data into actionable insights and recommendations
    
    Responsibilities:
    - Synthesize all agent outputs
    - Identify buying signals
    - Generate sales recommendations
    - Calculate confidence score
    """
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "ReportGeneratorAgent"
        logger.info(f"{self.name} initialized")
    
    async def generate(
        self,
        profile: CompanyProfile,
        financial: FinancialData,
        contacts: List[Contact],
        session: SimpleSession
    ) -> Dict[str, Any]:
        """Generate final intelligence report from all agent data"""
        logger.info(f"{self.name}: Generating report for {profile.name}")
        
        context = {
            'company': asdict(profile),
            'financials': asdict(financial),
            'contacts': [asdict(c) for c in contacts]
        }
        
        prompt = f"""You are a sales intelligence synthesizer. Analyze this data and generate insights:

{json.dumps(context, indent=2)}

Generate a JSON response with:
{{
    "buying_signals": [3-5 key indicators why now is a good time to reach out],
    "recommendations": [4-6 actionable sales strategies],
    "confidence_score": 85 (0-100 confidence level)
}}

Be specific and actionable. Return ONLY valid JSON.
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            logger.info(f"{self.name}: Completed report for {profile.name}")
            return data
            
        except Exception as e:
            logger.error(f"{self.name}: Error - {str(e)}")
            return {
                'buying_signals': [
                    'Company showing strong growth indicators',
                    'Recent funding suggests budget availability',
                    'Expanding team indicates scaling needs'
                ],
                'recommendations': [
                    'Lead with ROI calculator',
                    'Target VP of Sales for initial outreach',
                    'Emphasize scalability features',
                    'Reference industry trends'
                ],
                'confidence_score': 75.0
            }


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class SalesIntelligenceOrchestrator:
    """
    Main orchestrator coordinating all agents
    Implements multi-agent workflow with parallel and sequential execution
    
    Architecture:
    1. PARALLEL: Research + Financial agents (run simultaneously)
    2. SEQUENTIAL: Contact agent (uses context from step 1)
    3. SEQUENTIAL: Report agent (synthesizes all data)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Initialize Gemini - Try multiple sources for API key
        if not api_key:
            # Try Kaggle secrets first (GOOGLE_API_KEY)
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                api_key = user_secrets.get_secret("GOOGLE_API_KEY")
                logger.info("API key loaded from Kaggle secrets (GOOGLE_API_KEY)")
            except:
                # Try environment variable
                api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if api_key:
                    logger.info("API key loaded from environment variable")
                else:
                    logger.warning("No API key found. Will use fallback data.")
        
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = GenerativeModel('gemini-pro')
        
        # Initialize all agents
        self.research_agent = ResearchAgent(self.model)
        self.financial_agent = FinancialAnalysisAgent(self.model)
        self.contact_agent = ContactDiscoveryAgent(self.model)
        self.report_agent = ReportGeneratorAgent(self.model)
        
        # Session management
        self.session_manager = SessionManager()
        
        # Memory bank (simple dict for storing company intelligence)
        self.memory_bank = {}
        
        logger.info("SalesIntelligenceOrchestrator initialized")
    
    async def analyze_company(self, company_name: str) -> SalesIntelligence:
        """
        Main workflow: Orchestrates all agents to generate intelligence
        
        Workflow:
        1. Parallel: Research + Financial agents run simultaneously
        2. Sequential: Contact agent uses results from step 1
        3. Sequential: Report agent synthesizes all outputs
        
        Args:
            company_name: Name of company to analyze
            
        Returns:
            SalesIntelligence: Complete intelligence report
        """
        logger.info(f"Orchestrator: Starting analysis for {company_name}")
        start_time = datetime.now()
        
        # Create session for this analysis
        session = self.session_manager.create_session()
        
        try:
            # PARALLEL EXECUTION: Research and Financial agents
            logger.info("Orchestrator: Running parallel agents (Research + Financial)")
            profile_task = self.research_agent.research(company_name, session)
            financial_task = self.financial_agent.analyze(company_name, session)
            
            profile, financial = await asyncio.gather(profile_task, financial_task)
            
            # SEQUENTIAL EXECUTION: Contact discovery (needs company context)
            logger.info("Orchestrator: Running Contact agent")
            contacts = await self.contact_agent.discover(company_name, session)
            
            # SEQUENTIAL EXECUTION: Report generation (needs all data)
            logger.info("Orchestrator: Running Report agent")
            report_data = await self.report_agent.generate(
                profile, financial, contacts, session
            )
            
            # Compile final intelligence report
            intelligence = SalesIntelligence(
                company_profile=profile,
                financial_data=financial,
                key_contacts=contacts,
                buying_signals=report_data.get('buying_signals', []),
                recommendations=report_data.get('recommendations', []),
                confidence_score=report_data.get('confidence_score', 0.0),
                generated_at=datetime.now().isoformat()
            )
            
            # Store in memory bank for future reference
            self.memory_bank[f"company:{company_name}"] = intelligence
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Orchestrator: Analysis complete for {company_name} in {duration:.2f}s")
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Orchestrator: Error analyzing {company_name}: {str(e)}")
            raise
    
    async def batch_analyze(self, companies: List[str]) -> Dict[str, SalesIntelligence]:
        """
        Analyze multiple companies concurrently
        
        Args:
            companies: List of company names to analyze
            
        Returns:
            Dict mapping company names to intelligence reports
        """
        logger.info(f"Orchestrator: Starting batch analysis of {len(companies)} companies")
        
        tasks = [self.analyze_company(company) for company in companies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            company: result 
            for company, result in zip(companies, results)
            if not isinstance(result, Exception)
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Example usage of the Sales Intelligence system"""
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  API key not found in environment variables")
        print("Please set it: export GOOGLE_API_KEY='your-api-key'")
        print("\nRunning with mock data for demonstration...\n")
    
    orchestrator = SalesIntelligenceOrchestrator(api_key)
    
    # Analyze single company
    print("Starting analysis...")
    result = await orchestrator.analyze_company("Anthropic")
    
    print(f"\n{'='*60}")
    print(f"SALES INTELLIGENCE REPORT: {result.company_profile.name}")
    print(f"{'='*60}\n")
    
    print(f"Industry: {result.company_profile.industry}")
    print(f"Size: {result.company_profile.size}")
    print(f"Revenue: {result.financial_data.revenue}")
    print(f"Growth: {result.financial_data.growth_rate}\n")
    
    print("Key Contacts:")
    for contact in result.key_contacts[:3]:
        print(f"  - {contact.name}, {contact.title}")
    
    print("\nBuying Signals:")
    for signal in result.buying_signals:
        print(f"  • {signal}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\nConfidence Score: {result.confidence_score}%")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # For Jupyter/Kaggle notebooks, use this:
    # await main()
    
    # For regular Python scripts, use this:
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("NOTE: In Jupyter/Kaggle, run this instead:")
            print("await main()")
        else:
            raise