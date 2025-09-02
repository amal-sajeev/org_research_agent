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

# --- Structured Output Models (ENHANCED) ---
class PersonaSearchQuery(BaseModel):
    """Model representing a specific search query for customer persona research."""

    search_query: str = Field(
        description="A targeted query for persona research, focusing on user behavior, demographics, pain points, and decision-making factors."
    )
    research_focus: str = Field(
        description="The focus area: 'competitor_customers_segment', 'demographics_info', 'pain_points_behavior', or 'apollo_optimization'"
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
    keywords_positive: list[str] = Field(description="Keywords for positive targeting")
    keywords_negative: list[str] = Field(description="Keywords to exclude")
    pain_points: list[str] = Field(description="Key challenges and problems")
    current_solutions: list[str] = Field(description="Existing tools/solutions they use")


class PersonaDataCollection(BaseModel):
    """Model representing a collection of persona data."""
    
    personas: list[PersonaData] = Field(description="List of personas with Apollo.io relevant data")


class ApolloSearchParameters(BaseModel):
    """Apollo.io People Search API parameters - EXACT SCHEMA MATCH."""
    
    # Person-level filters - EXACT FIELD NAMES
    person_titles: list[str] = Field(default=[], description="Job titles to search for")
    person_seniorities: list[str] = Field(default=[], description="Seniority levels")
    q_keywords: list[str] = Field(default=[], description="Keywords associated with person")
    
    # Organization-level filters - EXACT FIELD NAMES
    organization_locations: list[str] = Field(default=[], description="Organization locations")
    q_organization_keywords: list[str] = Field(default=[], description="Organization keywords")
    
    # Contact data filters - EXACT FIELD NAMES
    email_status: list[str] = Field(default=['verified'], description="Email status filters")
    phone_status: list[str] = Field(default=['verified'], description="Phone status filters")

# --- Report-Based Context Extraction (NO SEARCH) ---
report_context_analyzer = LlmAgent(
    model=config.worker_model,  # Use lighter model for analysis
    name="report_context_analyzer",
    description="Analyzes provided market and segmentation reports to extract actionable context on product category, competitor landscape, key industries, buyer roles, company size patterns, and geographic markets, without external search.",
    instruction="""
    
    **INPUT DATA:**
    - user input : {input_analysis}
    - Segmentation Report: {segmentation_report}
    
    You are a strategic market analyst. Analyze the provided market_report and segmentation_report to extract actionable insights:
    
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
    - product_intent
    - competitors (list of strings) and target organization(if true)
    - industries (list of strings)
    - buyer_roles / segments (list of strings)
    - geographic_markets (list of strings)
    - priority_segments (list of objects with name, characteristics)
    """,
    output_key="report_context",
)

# # --- Enhanced Persona Intelligence Generator ---
# persona_intelligence_generator = LlmAgent(
#     model=config.search_model,
#     name="persona_intelligence_generator", 
#     description="Generates detailed persona intelligence with minimal targeted searches.",
#     planner=BuiltInPlanner(
#         thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
#     ),
#     instruction=f"""
#     You are a persona intelligence expert creating Apollo.io-optimized customer profiles.

#     **INPUT DATA:**
#     - Market Report: `{market_report}`
#     - Segmentation Report: `{segmentation_report}`
#     - Extracted Context: `{{report_context}}`
#     - User Input: `{{input_analysis}}`

#     **RESEARCH STRATEGY:**
    
#     **Phase 1: Report Analysis (Primary Source - 80%)**
#     Extract from provided reports:
#     - Specific buyer personas and decision-making roles
#     - Industry verticals and company characteristics  
#     - Geographic distribution patterns
#     - Technology adoption and solution preferences
#     - Competitor customer profiles

#     **Phase 2: Targeted Search (Gap Filling - 20%)**
#     Execute MAXIMUM 3-4 strategic searches ONLY for:
#     - Standard job title variations: "[product category] standard job titles 2024"
#     - Industry-specific roles: "[target industry] decision makers org chart"
#     - Apollo.io format verification: "Apollo.io job title best practices"

#     **PERSONA GENERATION REQUIREMENTS:**
#     Create 3-4 distinct personas with:

#     **High-Relevance Keywords Only:**
#     - job_titles: Industry-specific, decision-maker focused titles (avoid generic)
#     - keywords_positive: Role-specific, technology-specific, pain-point related terms
#     - keywords_negative: Clear exclusion terms to prevent irrelevant matches
#     - seniority_levels: Complete hierarchy coverage

#     **Keyword Quality Standards:**
#     - HIGHLY SPECIFIC to industry and role context
#     - DIRECTLY RELATED to product category and buyer behavior
#     - MEASURABLE impact on targeting precision
#     - NO GENERIC or AMBIGUOUS terms

#     **OUTPUT:** Complete PersonaDataCollection with validated, Apollo.io-ready targeting data.
#     """,
#     tools=[google_search],
#     # output_schema=PersonaDataCollection,
#     output_key="persona_data_collection",
# )

persona_intelligence_generator = LlmAgent(
    model=config.search_model,  # Only one search-model usage
    name="persona_intelligence_generator",
    description="Generates detailed persona intelligence using market reports and minimal targeted searches, focusing on relevant job titles, organization keywords, and target locations.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction=f"""
    You are a persona intelligence expert creating Apollo.io-optimized customer profiles based on the provided market and segmentation reports (and any available company/product information).

    **PRIMARY DATA SOURCES:**
    1. Market Report: `{{market_report}}`
    2. Segmentation Report: `{{segmentation_report}}`
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
    2. "[Key competitor] typical customer demographics"
    3. "[Primary industry] technology adoption survey recent"

    **APOLLO.IO OPTIMIZATION REQUIREMENTS:**
    Generate 3-4 distinct target personas with complete data for:

    **Person-Level Filters:**
    - person_titles: 10-15 specific job titles
    - person_seniorities: All levels (individual_contributor, manager, director, vp, c_suite)
    - q_keywords: Role-specific terms

    **Organization-Level Filters:**
    - q_organization_keywords: Company characteristics
    - organization_locations: Company target locations (City, State, Country)

    **Quality Standards:**
    - Prioritize report-based intelligence over web searches
    - Use searches only to fill further understandings of the product and to generator apporpriate keywords
    - Generate actionable, precise targeting parameters
    - Include both positive and negative targeting criteria
    - Exclude overly generic job titles and unrelated company tags

    Output complete PersonaDataCollection with all Apollo.io-ready data.
    """,
    tools=[google_search],
    output_key="persona_data_collection",
)


# --- FIXED Persona Quality Validator ---
persona_quality_validator = LlmAgent(
    model=config.critic_model,
    name="persona_quality_validator",
    description="Validates persona data quality with strict keyword relevance assessment.",
    instruction=f"""
    **INPUT DATA:**
    - Report Context: `{{report_context}}`
    - Persona Data Collection: `{{persona_data_collection}}`
    - User Input: `{{input_analysis}}`
    
    You are a strict persona data quality validator. Your job is to evaluate persona quality and provide a clear PASS/FAIL grade.

    **VALIDATION CHECKLIST:**

    **1. Keyword Relevance Assessment (CRITICAL):**
    - Are job_titles industry-specific and decision-maker focused?
    - Do keywords_positive directly relate to product category and buyer behavior?
    - Are keywords_negative effective at excluding irrelevant prospects?
    - Do all keywords align with market context from reports?

    **2. Persona Completeness:**
    - 3+ distinct, well-differentiated personas
    - 8+ specific job titles per persona category
    - Complete seniority coverage (individual_contributor through c_suite)
    - Clear pain points and current solutions

    **3. Market Context Alignment:**
    - Personas reflect priority segments from segmentation report
    - Geographic targeting matches market opportunities
    - Industry focus aligns with target verticals
    - Competitive landscape considerations included

    **STRICT GRADING CRITERIA:**

    **Grade "fail" if ANY of these exist:**
    - Generic job titles (e.g., "Manager", "Analyst" without context)
    - Ambiguous keywords that could apply to any industry
    - Missing industry-specific terminology
    - Poor alignment with report context
    - Insufficient exclusion criteria
    - Personas lack clear differentiation

    **Grade "pass" ONLY if:**
    - ALL keywords are highly relevant and specific
    - Job titles are industry-appropriate and decision-focused
    - Organization keywords target ideal customer characteristics
    - Geographic targeting is strategically aligned
    - Personas are well-differentiated and market-segment specific

    **FOLLOW-UP QUERY GENERATION (If "fail"):**
    Generate 3-5 specific searches targeting critical gaps:
    - "[industry] specific job titles Apollo targeting"
    - "[product category] buyer personas keywords 2024"
    - "[competitor] customer demographics profiles"

    **CRITICAL:** Be strict in evaluation. High keyword relevance is non-negotiable.

    **OUTPUT:** Single PersonaFeedback object with clear grade and specific feedback.
    """,
    output_schema=PersonaFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="persona_evaluation",
)

# --- FIXED Persona Escalation Checker ---
class PersonaEscalationChecker(BaseAgent):
    """Enhanced escalation checker with detailed logging and state validation."""
    
    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        try:
            # Get evaluation result from session state
            evaluation_result = ctx.session.state.get("persona_evaluation")
            
            logging.info(f"[{self.name}] Checking evaluation result: {evaluation_result}")
            
            if evaluation_result is None:
                logging.warning(f"[{self.name}] No persona_evaluation found in state")
                yield Event(author=self.name)
                return
                
            # Handle both dict and object access patterns
            grade = None
            if isinstance(evaluation_result, dict):
                grade = evaluation_result.get("grade")
            elif hasattr(evaluation_result, 'grade'):
                grade = evaluation_result.grade
            
            logging.info(f"[{self.name}] Extracted grade: {grade}")
            
            if grade == "pass":
                logging.info(f"[{self.name}] Persona evaluation PASSED. Escalating to stop loop.")
                yield Event(
                    author=self.name, 
                    actions=EventActions(escalate=True),
                    content="Persona validation passed - proceeding to parameter optimization."
                )
            else:
                logging.info(f"[{self.name}] Persona evaluation FAILED (grade: {grade}). Continuing loop for enhancement.")
                yield Event(
                    author=self.name,
                    content=f"Persona validation failed with grade: {grade}. Proceeding to enhancement phase."
                )
                
        except Exception as e:
            logging.error(f"[{self.name}] Error in escalation check: {e}")
            yield Event(author=self.name)

# --- ENHANCED Follow-up Research Agent ---
enhanced_persona_generator = LlmAgent(
    model=config.search_model,
    name="enhanced_persona_generator",
    description="Performs targeted follow-up research to address specific persona quality gaps.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a precision persona enhancement specialist addressing specific quality gaps.
    
    **INPUT DATA:**
    - Existing Persona Data: `{{persona_data_collection}}`
    - Quality Evaluation: `{{persona_evaluation}}`
    - Market Context: `{{report_context}}`
    
    **ENHANCEMENT MISSION:**
    The persona validation failed. Execute targeted research to address specific gaps identified in the evaluation feedback.
    
    **EXECUTION PROTOCOL:**
    
    1. **Gap Analysis**: Review the 'persona_evaluation' comment to identify specific deficiencies
    2. **Targeted Research**: Execute ALL follow_up_queries systematically
    3. **Keyword Refinement**: Focus on replacing generic/ambiguous terms with highly specific alternatives
    4. **Relevance Validation**: Ensure all new keywords directly relate to market context and user requirements
    
    **SEARCH OPTIMIZATION:**
    For each follow_up_query:
    - Use exact search terms with current year modifiers (2024, latest)
    - Focus on industry-specific terminology and role standardization
    - Cross-reference against competitor customer profiles
    - Validate geographic and demographic alignment
    
    **ENHANCEMENT FOCUS:**
    
    **Job Title Precision:**
    - Replace generic titles with industry-specific roles
    - Include both traditional and modern role variations
    - Cover complete decision-making hierarchy
    
    **Keyword Relevance:**
    - Remove ambiguous terms that could apply to any industry
    - Add industry-specific professional terminology
    - Include technology and solution-specific keywords
    - Strengthen exclusion criteria for negative targeting
    
    **Geographic Refinement:**
    - Align location targeting with market opportunities
    - specific geographic filters
    - Consider regional industry concentration patterns
    
    **CRITICAL OUTPUT REQUIREMENT:**
    Output enhanced PersonaDataCollection that specifically addresses evaluation gaps.
    - Maintain existing persona structure
    - Enhance keyword relevance and specificity
    - Remove or replace any generic/ambiguous terms
    - Ensure all targeting parameters align with market context
    
    **SUCCESS CRITERIA:**
    - All keywords are highly relevant to market context and personas
    - Job titles are industry-specific and decision-focused
    - Organization keywords target ideal customer characteristics precisely
    - Geographic targeting strategically aligned with opportunities
    - Exclusion criteria effectively filter irrelevant prospects
    """,
    tools=[google_search],
    output_key="persona_data_collection",
)

# --- FIXED Apollo Parameter Optimizer with Strict Schema Compliance ---
apollo_parameter_optimizer = LlmAgent(
    model=config.critic_model,
    name="apollo_parameter_optimizer",
    description="Generates EXACTLY the specified Apollo.io search parameters with strict schema compliance.",
    instruction=f"""
    **INPUT DATA:**
    - Persona Data Collection: `{{persona_data_collection}}`
    - Report Context: `{{report_context}}`

    **CRITICAL OUTPUT REQUIREMENT:** 
    Output ONLY the raw ApolloSearchParameters JSON object. NO explanations, NO additional text, NO quality reports.
    Include only the most important keywords for each field in terms of relevance
    
    **EXACT SCHEMA FIELDS (MANDATORY):**
    - person_titles: list[str]
    - person_seniorities: list[str] 
    - q_keywords: list[str]
    - organization_locations: list[str]
    - q_organization_keywords: list[str]  
    - email_status: list[str] 
    - phone_status: list[str]

    **PARAMETER MAPPING:**
    - person_titles: Extract job titles from all personas (industry-specific only)
    - person_seniorities: ["individual_contributor", "manager", "director", "vp", "c_suite", "owner", "partner", "founder"]
    - q_keywords: Combine keywords_positive from personas (highly relevant only)
    - organization_locations: Geographic markets from report_context
    - q_organization_keywords: Industry and company characteristics from personas (industry-specific keywords) (highly relevant only)
    - email_status: ["verified"]
    - phone_status: ["verified"]

    **FORBIDDEN OUTPUTS:**
    - NO explanatory text before or after the JSON
    - NO quality assurance reports
    - NO recommendations or next steps
    - NO additional fields beyond schema
    - NO comments within the JSON structure

    **MANDATORY OUTPUT FORMAT:**
    Return ONLY the JSON object that matches ApolloSearchParameters schema exactly.
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
    description="Retrieves previously generated market analysis and segmentation reports from state for downstream use.",
    instruction="Output: 'Market and segmentation reports retrieved successfully.'",
    after_agent_callback=[get_market_context],
    output_key="report_retrieval_status"
)

# --- Enhanced Prospect Research Pipeline ---
prospect_researcher = SequentialAgent(
    name="prospect_researcher",
    description="Executes comprehensive persona research with enhanced quality validation loop.",
    sub_agents=[
        retrieve_market_report,
        report_context_analyzer,
        persona_intelligence_generator,
        LoopAgent(
            name="persona_quality_assurance_loop",
            max_iterations=2,  # Increased iterations for quality assurance
            sub_agents=[
                persona_quality_validator,
                PersonaEscalationChecker(name="persona_escalation_checker"),
                enhanced_persona_generator,
            ],
        ),
        apollo_parameter_optimizer,
    ],
)
