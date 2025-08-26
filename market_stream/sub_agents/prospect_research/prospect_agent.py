import datetime
import logging
import re
from collections.abc import AsyncGenerator
from typing import Literal

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from google.adk.models import Gemini

from ...config import config

market_report = ""
segmentation_report = ""

# --- Structured Output Models (UNCHANGED) ---
class PersonaSearchQuery(BaseModel):
    """Model representing a specific search query for customer persona research."""

    search_query: str = Field(
        description="A targeted query for customer persona research, focusing on user behavior, demographics, pain points, and decision-making factors."
    )
    research_focus: str = Field(
        description="The focus area: 'competitor_customers', 'demographics_roles', 'pain_points_behavior', or 'apollo_optimization'"
    )


class PersonaFeedback(BaseModel):
    """Model for evaluating persona research quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if persona research is comprehensive with actionable Apollo.io filters, 'fail' if critical gaps exist."
    )
    comment: str = Field(
        description="Detailed evaluation focusing on persona depth, competitor coverage, and Apollo.io filter compatibility."
    )
    follow_up_queries: list[PersonaSearchQuery] | None = Field(
        default=None,
        description="Specific follow-up searches to fill persona research gaps, focusing on missing demographic data, competitor intelligence, or behavioral insights.",
    )


class PersonaData(BaseModel):
    """Simplified model for collecting persona data for Apollo.io parameter generation."""
    
    persona_name: str = Field(description="Descriptive name for this persona")
    job_titles: list[str] = Field(description="List of relevant job titles")
    seniority_levels: list[str] = Field(description="Seniority levels (director, vp, c_suite, etc.)")
    departments: list[str] = Field(description="Relevant departments")
    company_sizes: list[str] = Field(description="Company size ranges")
    industries: list[str] = Field(description="Relevant industries")
    locations: list[str] = Field(description="Geographic locations/regions")
    skills: list[str] = Field(description="Professional skills and competencies")
    keywords_positive: list[str] = Field(description="Keywords for positive targeting")
    keywords_negative: list[str] = Field(description="Keywords to exclude")
    company_keywords: list[str] = Field(description="Company-level keywords")
    technologies: list[str] = Field(description="Technologies and tools used")
    pain_points: list[str] = Field(description="Key challenges and problems")
    current_solutions: list[str] = Field(description="Existing tools/solutions they use")


class PersonaDataCollection(BaseModel):
    """Model representing a collection of persona data."""
    
    personas: list[PersonaData] = Field(description="List of personas with Apollo.io relevant data")


class ApolloSearchParameters(BaseModel):
    """Comprehensive Apollo.io People Search API parameters."""
    
    # Person-level filters
    person_titles: list[str] = Field(default=[], description="Job titles to search for")
    person_locations: list[str] = Field(default=[], description="Person locations (City, State, Country format)")
    person_seniorities: list[str] = Field(default=[], description="Seniority levels")
    person_departments: list[str] = Field(default=[], description="Department classifications")
    person_skills: list[str] = Field(default=[], description="Professional skills")
    person_keywords: list[str] = Field(default=[], description="Keywords associated with person")
    
    # Organization-level filters
    organization_locations: list[str] = Field(default=[], description="Organization locations")
    organization_num_employees_ranges: list[str] = Field(default=[], description="Employee count ranges")
    organization_industry_tag_ids: list[str] = Field(default=[], description="Industry classifications")
    organization_keywords: list[str] = Field(default=[], description="Organization keywords")
    organization_technologies: list[str] = Field(default=[], description="Technologies used by organizations")
    
    # Exclusion filters
    person_not_titles: list[str] = Field(default=[], description="Job titles to exclude")
    person_not_keywords: list[str] = Field(default=[], description="Keywords to exclude for persons")
    organization_not_keywords: list[str] = Field(default=[], description="Organization keywords to exclude")
    
    # Contact data filters
    email_status: list[str] = Field(default=[], description="Email status filters (verified, likely, etc.)")
    phone_status: list[str] = Field(default=[], description="Phone status filters")
    
    # Search configuration
    page: int = Field(default=1, description="Page number for pagination")
    per_page: int = Field(default=100, description="Results per page (max 100)")
    sort_by_field: str = Field(default="relevance", description="Field to sort results by")
    sort_ascending: bool = Field(default=False, description="Sort order")


# --- Report-Based Context Extraction (NO SEARCH) ---
report_context_analyzer = LlmAgent(
    model=config.worker_model,  # Use lighter model for analysis
    name="report_context_analyzer",
    description="Analyzes market and segmentation reports to extract key context without external search.",
    instruction=f"""
    You are a strategic report analyst. Analyze the provided market_report and segmentation_report to extract:

    **From Market Report:**
    - Product category and type
    - Primary competitor landscape
    - Key industry verticals
    - Geographic market presence
    - Technology ecosystem context

    **From Segmentation Report:**
    - Priority customer segments
    - Typical buyer personas and roles
    - Company size patterns
    - Industry distribution
    - Decision-making hierarchies

    **CRITICAL:** Base your analysis ONLY on the provided reports. Do not make assumptions or add external knowledge.

    Provide output as JSON with keys:
    - product_category (string)
    - competitors (list of strings)
    - industries (list of strings)
    - buyer_roles (list of strings)
    - company_sizes (list of strings)
    - geographic_markets (list of strings)
    - priority_segments (list of objects with name, characteristics)
    """,
    output_key="report_context",
)

# --- Consolidated Persona Generator (REDUCED SEARCH) ---
persona_intelligence_generator = LlmAgent(
    model=config.search_model,  # Only one search-model usage
    name="persona_intelligence_generator",
    description="Creates comprehensive persona intelligence with minimal targeted searches.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction=f"""
    You are a persona intelligence expert creating Apollo.io-optimized customer profiles.

    **PRIMARY DATA SOURCES:**
    1. Market Report: `{market_report}`
    2. Segmentation Report: `{segmentation_report}`
    3. Extracted Context: `{{report_context}}`

    **RESEARCH STRATEGY:**
    **Phase 1: Report Analysis (80% of intelligence)**
    Extract from provided reports:
    - Buyer personas and decision-making roles
    - Industry and company size patterns
    - Technology adoption and current solutions
    - Geographic distribution and market segments
    - Competitor customer characteristics

    **Phase 2: Targeted Search (20% - Only for critical gaps)**
    Execute MAXIMUM 3-5 strategic searches for:
    - Standard job title variations for identified roles
    - Apollo.io filter format specifications
    - Current industry technology adoption patterns

    **SEARCH FOCUS (If needed):**
    1. "[Identified product category] standard job titles Apollo.io format"
    2. "[Key competitor] typical customer demographics LinkedIn"
    3. "[Primary industry] technology adoption survey recent"

    **APOLLO.IO OPTIMIZATION REQUIREMENTS:**
    Generate 4-6 distinct personas with complete data for:
    
    **Person-Level Filters:**
    - person_titles: 15-25 specific job titles
    - person_seniorities: All levels (individual_contributor, manager, director, vp, c_suite)
    - person_departments: Relevant departments
    - person_locations: Geographic targeting
    - person_skills: Professional competencies
    - person_keywords: Role-specific terms

    **Organization-Level Filters:**
    - organization_num_employees_ranges: Company sizes (1-10, 11-50, 51-200, 201-1000, 1001-5000, 5001-10000, 10000+)
    - organization_industry_tag_ids: Industry classifications
    - organization_keywords: Company characteristics
    - organization_technologies: Technology stack indicators

    **Quality Standards:**
    - Prioritize report-based intelligence over web searches
    - Use searches only to fill specific Apollo.io format requirements
    - Generate actionable, precise targeting parameters
    - Include both positive and negative targeting criteria

    Output complete PersonaDataCollection with all Apollo.io-ready data.
    """,
    tools=[google_search],
    output_key="persona_data_collection",
)

# --- Quality Validator (NO SEARCH) ---
persona_quality_validator = LlmAgent(
    model=config.critic_model,
    name="persona_quality_validator",
    description="Validates persona completeness for Apollo.io without additional searches.",
    instruction=f"""
    You are a data quality expert validating Apollo.io persona completeness.

    **VALIDATION CRITERIA:**
    Assess PersonaDataCollection against Apollo.io requirements:

    **Critical Requirements (Must Pass):**
    - 4+ distinct personas covering different segments
    - 15+ total unique job titles across personas
    - Complete seniority level coverage (IC to C-suite)
    - Company size ranges in Apollo format
    - Industry classifications present
    - Geographic targeting data available
    - Professional skills and keywords defined

    **Quality Indicators:**
    - Personas differentiated by role, company size, industry
    - Technology adoption patterns identified
    - Pain points and current solutions documented
    - Exclusion criteria for negative targeting
    - Comprehensive department and skill coverage

    **GRADING:**
    - **Pass:** All critical requirements met, high-quality targeting data
    - **Fail:** Missing critical elements or insufficient data quality

    **NO SEARCH RECOMMENDATIONS:**
    If grading "fail", suggest specific data improvements based on existing reports, not new searches.

    Output single JSON PersonaFeedback object.
    """,
    output_schema=PersonaFeedback,
    output_key="persona_validation",
)

# --- Apollo Parameter Optimizer (NO SEARCH) ---
apollo_parameter_optimizer = LlmAgent(
    model=config.critic_model,
    name="apollo_parameter_optimizer",
    description="Generates optimized Apollo.io search parameters from persona data.",
    instruction=f"""
    You are an Apollo.io parameter optimization specialist.

    **INPUT DATA:**
    - Persona Data Collection: `{{persona_data_collection}}`
    - Market Context: `{{report_context}}`

    **CONSOLIDATION STRATEGY:**
    Create comprehensive Apollo.io People Search parameters by:

    **1. Person-Level Parameters:**
    - **person_titles:** All unique job titles from personas
    - **person_seniorities:** individual_contributor, manager, director, vp, c_suite, owner, partner, founder
    - **person_departments:** engineering, marketing, sales, finance, operations, hr, legal, etc.
    - **person_locations:** City, State, Country format from geographic data
    - **person_skills:** Professional skills and competencies
    - **person_keywords:** Positive targeting keywords

    **2. Organization-Level Parameters:**
    - **organization_num_employees_ranges:** ["1,10", "11,50", "51,200", "201,1000", "1001,5000", "5001,10000", "10000+"]
    - **organization_industry_tag_ids:** Industry classifications from personas
    - **organization_keywords:** Company-level targeting terms
    - **organization_technologies:** Technology stack indicators

    **3. Exclusion Parameters:**
    - **person_not_titles:** Irrelevant job titles to exclude
    - **person_not_keywords:** Negative person keywords
    - **organization_not_keywords:** Negative company keywords

    **4. Quality & Contact Filters:**
    - **email_status:** ["verified", "likely"]
    - **per_page:** 100
    - **sort_by_field:** "relevance"

    **OPTIMIZATION PRINCIPLES:**
    - Maximize relevant lead coverage without over-filtering
    - Balance precision with reach
    - Include comprehensive exclusion criteria
    - Optimize for lead qualification efficiency

    Generate single optimized ApolloSearchParameters object.
    """,
    output_schema=ApolloSearchParameters,
    output_key="prospect_researcher",
)

def get_market_context(callback_context: CallbackContext):
    try:
        callback_context.state["market_report"] = callback_context.state.get('market_intelligence_agent', '')
        callback_context.state["segmentation_report"] = callback_context.state.get('segmentation_intelligence_agent', '')
    except Exception as e:
        print(f"Error retrieving market context: {e}")

retrieve_market_report = LlmAgent(
    name="retrieve_market_report",
    model=config.worker_model,
    description="Retrieves market intelligence and segmentation reports from state.",
    instruction="Output: 'Market and segmentation reports retrieved successfully.'",
    after_agent_callback=[get_market_context],
    output_key="report_retrieval_status"
)

# --- OPTIMIZED PIPELINE (Removed Loop and Reduced Search Usage) ---
prospect_researcher = SequentialAgent(
    name="prospect_researcher",
    description="Efficiently generates Apollo.io search parameters with minimal external searches.",
    sub_agents=[
        retrieve_market_report,
        report_context_analyzer,  # NO SEARCH - Analyzes existing reports
        persona_intelligence_generator,  # MINIMAL SEARCH - Only 3-5 targeted searches
        persona_quality_validator,  # NO SEARCH - Validates completeness
        apollo_parameter_optimizer,  # NO SEARCH - Final parameter generation
    ],
)

# # --- MAIN STREAMLINED APOLLO AGENT ---
# prospect_researcher = LlmAgent(
#     name="prospect_researcher",
#     model = config.worker_model,
#     description="Streamlined customer persona research assistant that generates consolidated Apollo.io search parameters through focused market analysis.",
#     instruction=f"""
#     You are a streamlined Apollo.io Lead Generation Assistant focused on creating optimized search parameters.

#     **CORE MISSION:**
#     Generate consolidated Apollo.io People Search API parameters through efficient persona research.

#     **REQUIRED INPUTS:**
#     Users must provide:
#     1. **Product/Service Description:** Features, benefits, use cases, target market
#     2. **Company Information:** Background, market positioning, competitive context
#     3. **Known Competitors (Optional):** Any known competitors (I'll research more)
#     4. **Market Analysis Report:** `{market_report}`
#     5. **Segmentation Analysis Report:** `{segmentation_report}`

#     **STREAMLINED WORKFLOW:**
#     You use an efficient research methodology with single quality review:

#     **Phase 1: Comprehensive Research**
#     - Identify job titles, seniority levels, and role variations
#     - Research company size patterns, industries, and geographic distribution
#     - Analyze professional skills, technology adoption, and behavioral patterns
#     - Map competitive landscape and alternative solution usage

#     **Phase 2: Quality Validation**
#     - Ensure sufficient data for effective Apollo.io parameter generation
#     - Validate job title precision and demographic coverage
#     - Confirm skills intelligence and technology adoption insights
#     - Execute follow-up searches to fill any critical gaps

#     **Phase 3: Parameter Generation**
#     - Transform research into structured persona data
#     - Consolidate all insights into unified Apollo.io search parameters
#     - Optimize for maximum relevant lead discovery
#     - Balance precision with comprehensive market coverage

#     **FINAL OUTPUT:**
#     Single JSON object with consolidated Apollo.io People Search API parameters:

#     ```json
#     {{
#       "person_titles": ["all relevant job titles"],
#       "person_seniorities": ["seniority levels"],
#       "person_locations": ["geographic targeting"],
#       "person_skills": ["professional skills"],
#       "person_keywords": ["positive keywords"],
#       "organization_num_employees_ranges": ["company sizes"],
#       "organization_industry_tag_ids": ["industries"],
#       "organization_keywords": ["company keywords"],
#       "organization_technologies": ["technology stack"],
#       "person_not_keywords": ["exclusion terms"],
#       "email_status": ["verified", "likely"],
#       "per_page": 100,
#       "sort_by_field": "relevance"
#     }}
#     ```

#     **KEY ADVANTAGES:**
#     - **Maximized Lead Volume:** Consolidates all personas into a single comprehensive search
#     - **Maintained Relevance:** Uses exclusion filters and qualification criteria
#     - **Immediate Implementation:** Direct API parameters for instant lead generation
#     - **Optimized Coverage:** Balances specificity with market opportunity

#     **RESEARCH FOCUS:**
#     All research specifically targets Apollo.io parameter optimization:
#     - Exact job titles suitable for person_titles filtering
#     - Professional skills and keywords for precise targeting
#     - Company characteristics for organization-level filtering
#     - Technology adoption patterns for intent signaling
#     - Geographic and industry targeting for market focus

#     Once you provide your product and company details (along with the market_report and segmentation_report reports), I will execute the streamlined research process and deliver your optimized Apollo.io search parameters as a single JSON object.

#     Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
#     Ready to generate your Apollo.io search parameters.
#     """,
#     sub_agents=[prospect_research_pipeline],
#     output_key="prospect_researcher",
# )
# root_agent = prospect_research_pipeline
