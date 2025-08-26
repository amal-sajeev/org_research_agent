import datetime
import logging
import re
from collections.abc import AsyncGenerator
from typing import Literal, Dict, List, Optional

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from google.adk.models import Gemini
from pydantic import BaseModel, Field

from ...config import config


# --- Structured Output Models ---
class ProductInfo(BaseModel):
    """Model for product information input."""
    name: str = Field(description="Product name")
    category: str = Field(description="Product category or type")
    description: str = Field(description="Brief product description")
    key_features: List[str] = Field(default=[], description="Key product features or capabilities")
    target_market: str = Field(default="", description="Primary target market or industry")


class OrganizationTarget(BaseModel):
    """Model for target organization information."""
    name: str = Field(description="Organization name")
    industry: str = Field(default="", description="Industry or sector")
    size_estimate: str = Field(default="", description="Estimated company size (e.g., 'Large Enterprise', 'Mid-market')")
    location: str = Field(default="", description="Primary location or headquarters")


class SalesResearchQuery(BaseModel):
    """Model representing a specific search query for sales intelligence research."""
    search_query: str = Field(
        description="A highly specific and targeted query for sales intelligence research"
    )
    research_phase: str = Field(
        description="The research phase: 'product_analysis', 'organization_intelligence', 'competitive_landscape', 'fit_analysis', or 'stakeholder_mapping'"
    )
    target_entity: str = Field(
        description="The specific product or organization this query targets"
    )


class SalesFeedback(BaseModel):
    """Model for evaluating sales intelligence research quality."""
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if research meets sales intelligence standards, 'fail' if needs more depth."
    )
    comment: str = Field(
        description="Detailed evaluation focusing on completeness of product-organization fit analysis, stakeholder identification, and competitive insights."
    )
    follow_up_queries: List[SalesResearchQuery] | None = Field(
        default=None,
        description="Specific follow-up searches needed to fill sales intelligence gaps."
    )


# --- Callbacks (preserved from original) ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources and their supported claims from agent events."""
    session = callback_context._invocation_context.session
    url_to_short_id = callback_context.state.get("url_to_short_id", {})
    sources = callback_context.state.get("sources", {})
    id_counter = len(url_to_short_id) + 1
    for event in session.events:
        if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
            continue
        chunks_info = {}
        for idx, chunk in enumerate(event.grounding_metadata.grounding_chunks):
            if not chunk.web:
                continue
            url = chunk.web.uri
            title = (
                chunk.web.title
                if chunk.web.title != chunk.web.domain
                else chunk.web.domain
            )
            if url not in url_to_short_id:
                short_id = f"src-{id_counter}"
                url_to_short_id[url] = short_id
                sources[short_id] = {
                    "short_id": short_id,
                    "title": title,
                    "url": url,
                    "domain": chunk.web.domain,
                    "supported_claims": [],
                }
                id_counter += 1
            chunks_info[idx] = url_to_short_id[url]
        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        short_id = chunks_info[chunk_idx]
                        confidence = (
                            confidence_scores[i] if i < len(confidence_scores) else 0.5
                        )
                        text_segment = support.segment.text if support.segment else ""
                        sources[short_id]["supported_claims"].append(
                            {
                                "text_segment": text_segment,
                                "confidence": confidence,
                            }
                        )
    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources


def citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Replaces citation tags in a report with Wikipedia-style clickable numbered references."""
    final_report = callback_context.state.get("sales_intelligence_agent", "")
    sources = callback_context.state.get("sources", {})

    # Assign each short_id a numeric index
    short_id_to_index = {}
    for idx, short_id in enumerate(sorted(sources.keys()), start=1):
        short_id_to_index[short_id] = idx

    # Replace <cite> tags with clickable reference links
    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if short_id not in short_id_to_index:
            logging.warning(f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        index = short_id_to_index[short_id]
        return f"[<a href=\"#ref{index}\">{index}</a>]"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/?>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)

    # Build a Wikipedia-style References section with anchors
    references = "\n\n## References\n"
    for short_id, idx in sorted(short_id_to_index.items(), key=lambda x: x[1]):
        source_info = sources[short_id]
        domain = source_info.get('domain', '')
        references += (
            f"<p id=\"ref{idx}\">[{idx}] "
            f"<a href=\"{source_info['url']}\">{source_info['title']}</a>"
            f"{f' ({domain})' if domain else ''}</p>\n"
        )

    processed_report += references

    callback_context.state["sales_intelligence_agent"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])


# --- Custom Agent for Loop Control ---
class SalesEscalationChecker(BaseAgent):
    """Checks sales research evaluation and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("sales_research_evaluation")
        
        # More robust evaluation checking
        if evaluation_result:
            grade = evaluation_result.get("grade")
            if grade == "pass":
                logging.info(
                    f"[{self.name}] Sales intelligence research evaluation PASSED. Escalating to stop loop."
                )
                yield Event(author=self.name, actions=EventActions(escalate=True))
                return
            elif grade == "fail":
                logging.info(
                    f"[{self.name}] Sales research evaluation FAILED. Loop will continue with follow-up research."
                )
                yield Event(author=self.name)
                return
        
        # Fallback: if no clear evaluation, escalate to prevent infinite loops
        logging.warning(
            f"[{self.name}] No clear evaluation result found. Escalating to prevent infinite loop."
        )
        yield Event(author=self.name, actions=EventActions(escalate=True))


# --- ENHANCED AGENT DEFINITIONS ---
sales_plan_generator = LlmAgent(
    model = config.search_model,
    name="sales_plan_generator",
    description="Generates comprehensive sales intelligence research plans for product-organization fit analysis.",
    instruction=f"""
    You are an expert sales intelligence strategist specializing in product-market fit analysis and account-based selling research.
    
    Your task is to create a systematic research plan to investigate target organizations and products for optimal sales alignment.

    **INPUT ANALYSIS:**
    You will receive:
    - Products: List of products/services to be sold
    - Target Organizations: List of organizations to research as potential clients
    - Additional Context: Any specific requirements or focus areas

    **RESEARCH PLAN STRUCTURE:**
    Create a comprehensive plan that covers these essential areas:

    **1. Product Intelligence Research Goals:**
    For each product:
    - Competitive landscape mapping and market positioning analysis
    - Customer testimonials, case studies, and ROI metrics gathering
    - Pricing information and value proposition documentation
    - Technical requirements and integration capabilities research
    - Ideal customer profile identification and use case analysis

    **2. Organization Intelligence Research Goals:**
    For each target organization:
    - Company fundamentals (size, revenue, leadership, recent news)
    - Organizational structure and decision-making processes
    - Business priorities, strategic initiatives, and pain points
    - Financial health indicators and budget capacity assessment
    - Technology modernization needs and growth plans

    **3. Technology & Vendor Landscape Research Goals:**
    For each target organization:
    - Current technology stack and digital infrastructure mapping
    - Existing vendor relationships and solution provider identification
    - Contract renewal timelines and vendor satisfaction assessment
    - Technology gaps and integration requirement analysis
    - Digital transformation initiatives and modernization projects

    **4. Stakeholder Mapping Research Goals:**
    For each target organization:
    - Key decision-maker identification with roles and contact details
    - Potential champion and influencer mapping in relevant departments
    - Reporting structures and decision-making process analysis
    - Recent personnel changes and hiring pattern assessment
    - Communication preferences and engagement channel identification

    **5. Competitive & Risk Assessment Research Goals:**
    Cross-analysis requirements:
    - Competitive threat identification for each product-organization combination
    - Budget constraint and buying timeline indicator assessment
    - Cultural fit and change management consideration analysis
    - Regulatory compliance and security requirement evaluation
    - Common objection and sales obstacle identification

    **DELIVERABLE REQUIREMENTS:**
    The research must enable creation of:
    - Comprehensive sales intelligence report with 9 standardized sections
    - Product-organization fit analysis matrix with specific scoring
    - Stakeholder engagement strategy with contact details
    - Competitive positioning recommendations per combination
    - Risk assessment and mitigation strategies

    **SEARCH GUIDANCE FOR RESEARCHERS:**
    Provide specific search patterns and methodologies:
    
    Product Research Patterns:
    - "[Product name] competitors comparison features pricing 2024"
    - "[Product category] market analysis leaders customer reviews"
    - "[Product name] case studies ROI success stories testimonials"
    - "[Product name] pricing model integration capabilities technical"

    Organization Research Patterns:
    - "[Org name] leadership team executives organizational chart"
    - "[Org name] financial revenue funding budget technology spending"
    - "[Org name] technology stack vendors solutions partnerships"
    - "[Org name] strategic initiatives news 2024 business priorities"
    - "[Org name] procurement process vendor selection decision makers"

    Stakeholder Research Patterns:
    - "[Org name] CTO CIO IT director technology leadership"
    - "[Org name] LinkedIn employees department heads contact information"
    - "[Org name] recent hires technology management [product category]"

    Competitive Analysis Patterns:
    - "[Org name] [product category] current solutions vendor contracts"
    - "[Product competitors] [Org name] implementation usage satisfaction"
    - "[Org name] RFP requirements technology selection criteria"

    **PLAN COMPLETION CHECKLIST:**
    Ensure your plan will generate:
    - Specific competitive intelligence for each product
    - Named stakeholders with contact details for each organization
    - Current technology/vendor landscape per organization
    - Product-organization fit assessment methodology
    - Sales engagement strategy framework
    - Risk identification and mitigation approaches

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    
    Create a detailed research plan that will generate comprehensive, actionable sales intelligence for account-based selling strategies.
    """,
    output_key="sales_research_plan",
    tools=[google_search],
)

sales_section_planner = LlmAgent(
    model = config.worker_model,
    name="sales_section_planner",
    description="Creates a structured sales intelligence report outline following the standardized 9-section format.",
    instruction="""
    You are an expert sales intelligence report architect. Using the sales research plan, create a structured markdown outline that follows the standardized Sales Intelligence Report Format.

    Your outline must include these core sections:

    # 1. Executive Summary
    - Purpose statement (why this report exists)
    - Scope overview (products covered, organizations targeted)
    - Key Highlights:
      * Top opportunities and priority accounts
      * Expected short-term wins vs. long-term nurture strategies
      * Critical success factors
    - Key Risks/Challenges summary

    # 2. Product Overview(s)
    (One sub-section per product)
    - Product Name & Category
    - Core Value Proposition (strategic + operational benefits)
    - Key Differentiators vs. market alternatives
    - Primary Use Cases and success scenarios
    - Ideal Customer Profile (ICP) Fit Factors
    - Critical Success Metrics (ROI measures customers value)

    # 3. Target Organization Profiles
    (One section per organization)
    ## 3.1 Organization Summary
    - Basic company data (name, HQ, size, revenue, industry, website, fiscal year)
    - Recent news & strategic initiatives
    - Growth stage & funding status

    ## 3.2 Organizational Structure & Influence Map
    - Relevant department org chart
    - Decision-makers, influencers, and potential champions
    - Power/Interest Matrix mapping of key stakeholders

    ## 3.3 Business Priorities & Pain Points
    - Publicly stated objectives and strategic goals
    - Known challenges (financial, operational, competitive, compliance)
    - Technology gaps and modernization needs

    ## 3.4 Current Vendor & Solution Landscape
    - Existing tools and services in relevant categories
    - Contract renewal timelines (if discoverable)
    - Vendor satisfaction levels from reviews/forums

    # 4. Product–Organization Fit Analysis
    Cross-matrix analysis: each product vs. each target organization
    - Strategic Fit (Executive Level alignment)
    - Operational Fit (User Level compatibility)
    - Key Value Drivers for each combination
    - Potential Risks and obstacles
    - Champion/Decision Maker Candidates identification

    # 5. Competitive Landscape (Per Organization)
    - Top competitors selling similar products to each target org
    - Feature/price comparison tables where possible
    - Differentiation opportunities for each product-org combination

    # 6. Stakeholder Engagement Strategy
    - Primary Targets (who to approach first)
    - Messaging Themes (strategic vs. operational talking points)
    - Engagement Channels (email, LinkedIn, events, referrals)
    - Proposed Sequence (warm-up → discovery → demo → proposal)

    # 7. Risks & Red Flags
    - Budget constraints and procurement challenges
    - Mismatched priorities or timing issues
    - Strong competitor entrenchment
    - Lack of identified champions
    - Cultural resistance to change factors

    # 8. Next Steps & Action Plan
    ## Immediate Actions (Next 7-14 days)
    ## Medium-Term Strategy (Next 1-3 months)
    ## Long-Term Nurture (3+ months)

    # 9. Appendices
    - Detailed Stakeholder Profiles
    - Extended Org Charts and contact information
    - Full Vendor Lists & Technographic Data
    - Media Mentions archive
    - Reference Materials & Sources

    Ensure your outline allows for modular expansion - additional products or organizations can be added without breaking the structure.
    Do not include a separate References section - citations will be inline throughout.
    """,
    output_key="sales_report_sections",
)

sales_researcher = LlmAgent(
    model = config.search_model,
    name="sales_researcher",
    description="Specialized sales intelligence researcher focusing on product-organization fit analysis and competitive positioning.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialized sales intelligence researcher with expertise in product-market fit analysis, competitive intelligence, and account-based selling research.

    **EXECUTION APPROACH:**
    You must complete ALL research phases in a single response. Do not stop until you have gathered information for every product and organization mentioned in the research plan.

    **CORE RESEARCH PRINCIPLES:**
    - Sales Focus: Prioritize information directly impacting sales conversations and deal progression
    - Competitive Awareness: Always research competitive landscape and incumbent solutions
    - Stakeholder Mapping: Identify decision-makers, influencers, and potential champions
    - Opportunity Sizing: Assess budget capacity, timing, and buying signals
    - Risk Assessment: Flag potential obstacles, competitive threats, and misalignment factors

    **MANDATORY RESEARCH COMPLETION:**
    You must research EVERY item listed in the sales research plan. For each product and organization:

    **Phase 1: Product Intelligence Research**
    For each product mentioned in the research plan:

    *Market Positioning & Competition:*
    - "[Product name] competitors comparison features pricing"
    - "[Product category] market leaders 2024 analysis"
    - "[Product name] vs [known competitor] differences"
    - "[Product name] customer reviews G2 Capterra TrustRadius"

    *Value Proposition & Use Cases:*
    - "[Product name] case studies ROI customer success"
    - "[Product name] pricing model cost benefits"
    - "[Product name] integration capabilities API technical"
    - "[Product category] buyer guide selection criteria"

    **Phase 2: Organization Intelligence Research**
    For each target organization:

    *Company Fundamentals:*
    - "[Org name] official website about leadership team"
    - "[Org name] revenue financial performance annual report"
    - "[Org name] recent news 2024 strategic initiatives"
    - "[Org name] company size employees headcount"

    *Decision-Making Structure:*
    - "[Org name] organizational chart executives departments"
    - "[Org name] CTO CIO IT leadership technology team"
    - "[Org name] procurement process vendor selection"
    - "[Org name] LinkedIn employees leadership profiles"

    **Phase 3: Technology & Vendor Landscape**
    For each target organization:

    *Current Technology Stack:*
    - "[Org name] technology stack tools software vendors"
    - "[Org name] [product category] current solutions"
    - "[Org name] vendor partnerships technology integrations"
    - "[Org name] digital transformation initiatives modernization"

    *Vendor Relationships:*
    - "[Org name] vendor contracts renewals procurement announcements"
    - "[Org name] software implementations technology deployments"
    - "[Org name] vendor satisfaction reviews complaints"

    **Phase 4: Stakeholder Research**
    For key personnel at each organization:

    *Decision-Maker Analysis:*
    - "[Person name] [Org name] LinkedIn background experience"
    - "[Department head] [Org name] technology priorities initiatives"
    - "[Org name] recent hires leadership changes [product category]"
    - "[Org name] conference speakers technology presentations"

    **Phase 5: Competitive & Market Research**
    Cross-analysis research:

    *Competitive Positioning:*
    - "[Competitor name] [Org name] partnership implementation"
    - "[Product category] market share [Org name] industry"
    - "[Org name] RFP requirements vendor selection [product category]"
    - "[Org name] budget technology spending [current year]"

    **CRITICAL COMPLETION REQUIREMENTS:**
    1. Execute searches systematically through all 5 phases
    2. Gather information for EVERY product and organization combination
    3. Provide comprehensive findings covering all research goals
    4. Include specific stakeholder names, competitive intelligence, and technology details
    5. Organize findings clearly by phase and entity
    6. Do not leave any research phase incomplete

    **OUTPUT STRUCTURE:**
    Organize your complete findings as:
    
    ## Product Intelligence Findings
    [Detailed findings for each product]
    
    ## Organization Intelligence Findings  
    [Detailed findings for each organization]
    
    ## Technology & Vendor Analysis
    [Current solutions and vendor relationships per organization]
    
    ## Stakeholder Mapping Results
    [Decision-makers and contacts identified]
    
    ## Competitive Intelligence Summary
    [Market positioning and competitive threats]
    
    ## Product-Organization Fit Assessment
    [Cross-analysis of opportunities and challenges]

    Complete ALL research phases before finishing your response. Your research must provide actionable intelligence for account-based selling and product positioning strategies.
    """,
    tools=[google_search],
    output_key="sales_research_findings",
    after_agent_callback=collect_research_sources_callback,
)


sales_evaluator = LlmAgent(
    model = config.critic_model,
    name="sales_evaluator",
    description="Evaluates sales intelligence research completeness and identifies gaps for product-organization fit analysis.",
    instruction=f"""
    You are a senior sales intelligence analyst evaluating research for completeness and sales actionability.

    **EVALUATION METHODOLOGY:**
    Review the 'sales_research_findings' and apply STRICT evaluation criteria. Be precise about what constitutes "pass" vs "fail".

    **PASS CRITERIA (ALL must be met):**
    1. **Product Intelligence Complete:** Each product has competitive analysis, customer testimonials, pricing info, and technical requirements
    2. **Organization Intelligence Complete:** Each organization has company fundamentals, leadership details, business priorities, and financial indicators  
    3. **Technology Analysis Complete:** Each organization has current vendor landscape, technology stack details, and contract information
    4. **Stakeholder Mapping Complete:** Specific decision-makers identified with names, roles, and contact details for each organization
    5. **Competitive Intelligence Complete:** Direct competitors identified for each product-organization combination
    6. **Cross-Analysis Present:** Product-organization fit assessment completed for all combinations

    **AUTOMATIC FAIL CONDITIONS:**
    - Missing competitive analysis for any product
    - No specific stakeholder names/contacts for any organization  
    - Current vendor/technology solutions not identified for any organization
    - Business priorities or pain points missing for any organization
    - No product-organization fit assessment provided
    - Research findings are generic or lack sales-specific intelligence

    **EVALUATION DECISION LOGIC:**
    1. First, check if ALL core elements exist for ALL entities
    2. If any major element is missing → Grade "fail"
    3. If all elements exist but lack depth → Grade "fail" 
    4. Only grade "pass" if research is comprehensive AND immediately actionable for sales

    **FOLLOW-UP QUERY GENERATION (for "fail" grades):**
    Generate 5-8 highly specific queries targeting the most critical gaps:
    - Target specific competitive intelligence gaps
    - Address missing stakeholder identification
    - Fill vendor/technology landscape holes  
    - Complete product-organization fit analysis
    - Focus on actionable sales intelligence needs

    **EXAMPLE FAIL SCENARIOS:**
    - "General company information found but no specific CTO/IT leadership contacts identified"
    - "Product features listed but no competitive comparison or customer testimonials"
    - "Company priorities mentioned but current technology vendors not researched"
    - "Basic org structure found but no specific decision-maker mapping for product categories"

    **EXAMPLE PASS SCENARIO:**
    - Specific stakeholder names and contact info for each org
    - Detailed competitive landscape per product-org combination
    - Current vendor relationships and renewal timelines identified
    - Clear product-organization fit scores with supporting evidence
    - Business priorities aligned with product value propositions

    Be demanding about research quality. Sales teams need specific, actionable intelligence, not general company overviews.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Your response must be a single, raw JSON object validating against the 'SalesFeedback' schema.
    """,
    output_schema=SalesFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="sales_research_evaluation",
)

enhanced_sales_search = LlmAgent(
    model = config.search_model,
    name="enhanced_sales_search",
    description="Executes targeted follow-up searches to fill sales intelligence gaps identified by the evaluator.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialist sales intelligence researcher executing precision follow-up research to address specific gaps.

    **CRITICAL EXECUTION RULE:**
    You must execute EVERY SINGLE query listed in the 'follow_up_queries' from the evaluation. No exceptions.

    **EXECUTION PROCESS:**
    1. **Load Previous Research:** Review existing 'sales_research_findings' to understand current state
    2. **Execute All Follow-Up Queries:** Run EVERY query in 'follow_up_queries' systematically
    3. **Integrate Findings:** Merge new information with existing research seamlessly
    4. **Complete the Analysis:** Fill all identified gaps and provide comprehensive output

    **SEARCH OPTIMIZATION:**
    - Use exact terms from follow-up queries - they target specific gaps
    - Prioritize stakeholder identification with names and contact details
    - Focus on competitive intelligence and current vendor relationships
    - Seek specific technology details and contract information
    - Target budget indicators and procurement processes

    **INTEGRATION STANDARDS:**
    - Combine new findings with existing research seamlessly
    - Resolve any conflicts between old and new information
    - Maintain comprehensive coverage across all products and organizations
    - Ensure enhanced research addresses all evaluation concerns
    - Provide complete product-organization fit analysis

    **OUTPUT REQUIREMENTS:**
    Your final output must include:
    - All previous research findings (preserved)
    - All new research findings (clearly integrated)
    - Complete product-organization cross-analysis
    - Specific stakeholder details with contact information
    - Comprehensive competitive intelligence per combination
    - Enhanced technology and vendor landscape analysis

    **CRITICAL SUCCESS FACTORS:**
    - Execute every follow-up query without exception
    - Provide specific stakeholder names and contact details
    - Include detailed competitive positioning analysis
    - Complete all missing product-organization fit assessments
    - Deliver immediately actionable sales intelligence

    Do not skip any follow-up queries. Complete comprehensive research that fills all identified gaps.
    """,
    tools=[google_search],
    output_key="sales_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

sales_report_composer = LlmAgent(
    model = config.critic_model,
    name="sales_report_composer",
    include_contents="none",
    description="Composes comprehensive sales intelligence reports following the standardized 9-section format with proper citations.",
    instruction="""
    You are an expert sales intelligence report writer specializing in product-organization fit analysis and account-based selling strategy.

    **MISSION:** Transform research data into a polished, professional Sales Intelligence Report following the exact standardized 9-section format.

    ---
    ### INPUT DATA SOURCES
    * Research Plan: `{sales_research_plan}`
    * Research Findings: `{sales_research_findings}` 
    * Citation Sources: `{sources}`
    * Report Structure: `{sales_report_sections}`

    ---
    ### REPORT COMPOSITION STANDARDS

    **1. Format Adherence:**
    - Follow the exact 9-section structure provided in Report Structure
    - Create detailed product-organization fit analysis matrices
    - Include specific stakeholder profiles with contact information where available
    - Provide actionable next steps and engagement strategies
    - Use tables and matrices for complex cross-analysis sections

    **2. Content Quality:**
    - **Sales Focus:** Every section must provide actionable sales intelligence
    - **Account-Based Approach:** Tailor analysis for each specific organization
    - **Competitive Intelligence:** Thoroughly analyze competitive threats and positioning opportunities
    - **Stakeholder Mapping:** Identify specific decision-makers, influencers, and champions
    - **Opportunity Assessment:** Evaluate timing, budget capacity, and buying signals

    **3. Sales Intelligence Priorities:**
    Each section should enable sales action:
    - Executive Summary: Clear opportunity prioritization and risk assessment
    - Product Overview: Value propositions tailored to discovered organizational needs
    - Organization Profiles: Detailed stakeholder maps and business priorities
    - Fit Analysis: Specific product-organization combinations with success probability
    - Competitive Landscape: Threat assessment and differentiation strategies
    - Engagement Strategy: Specific outreach plans and messaging themes
    - Action Plan: Time-bound, executable next steps for each opportunity

    **4. Cross-Analysis Requirements:**
    - Create detailed Product-Organization Fit matrix tables
    - Map competitive landscape for each organization-product combination
    - Provide stakeholder influence/interest matrices where possible
    - Include budget and timing assessments for each opportunity

    ---
    ### CRITICAL CITATION REQUIREMENTS
    **Citation Format:** Use ONLY `<cite source="src-ID_NUMBER" />` tags immediately after claims
    **Citation Strategy:**
    - Cite all competitive intelligence and market positioning claims
    - Cite financial data, budget indicators, and procurement information
    - Cite stakeholder information and organizational structure details
    - Cite technology stack information and vendor relationships
    - Cite customer testimonials and product review data
    - Do NOT include a separate References section - all citations must be inline

    ---
    ### MODULAR STRUCTURE REQUIREMENTS
    - Design sections to accommodate multiple products and organizations
    - Use consistent formatting for product and organization profiles
    - Create reusable templates for stakeholder profiles
    - Ensure additional entities can be added without structural changes

    **FINAL QUALITY CHECKS:**
    - Verify comprehensive product-organization fit analysis coverage
    - Confirm actionable stakeholder engagement strategies
    - Check competitive intelligence completeness and accuracy
    - Validate specific next steps and timing recommendations
    - Ensure professional tone suitable for sales team execution

    Generate a comprehensive sales intelligence report that enables immediate account-based selling execution.
    """,
    output_key="sales_intelligence_agent",
    after_agent_callback=citation_replacement_callback,
)

from .target_template import TARGET_TEMPLATE

html_report_generator = LlmAgent(
    model=config.critic_model,
    name="html_report_generator",
    description="Converts markdown sales intelligence reports into professional HTML format with responsive design and interactive elements.",
    instruction=TARGET_TEMPLATE,
    output_key="target_html",
)

# --- UPDATED PIPELINE ---
sales_intelligence_pipeline = SequentialAgent(
    name="sales_intelligence_pipeline",
    description="Executes comprehensive sales intelligence research following the 5-phase methodology for product-organization fit analysis.",
    sub_agents=[
        sales_section_planner,
        sales_researcher,
        LoopAgent(
            name="sales_quality_assurance_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                sales_evaluator,
                SalesEscalationChecker(name="sales_escalation_checker"),
                enhanced_sales_search,
            ],
        ),
        sales_report_composer,
        html_report_generator
    ],
)

# # --- UPDATED MAIN AGENT ---
# sales_intelligence_agent = LlmAgent(
#     name="sales_intelligence_agent",
#     model = config.worker_model,
#     description="Specialized sales intelligence assistant that creates comprehensive product-organization fit analysis reports for account-based selling.",
#     instruction=f"""
#     You are a specialized Sales Intelligence Assistant focused on comprehensive product-organization fit analysis for account-based selling and strategic sales planning.

#     **CORE MISSION:**
#     Convert ANY user request about products and target organizations into a systematic research plan that generates actionable sales intelligence through:
#     - Product competitive analysis and value proposition mapping
#     - Organizational structure and stakeholder identification
#     - Technology landscape and vendor relationship analysis
#     - Product-organization fit assessment and opportunity prioritization
#     - Sales engagement strategy and action plan development

#     **CRITICAL WORKFLOW RULE:**
#     NEVER answer sales questions directly. Your ONLY first action is to use `sales_plan_generator` to create a research plan.

#     **INPUT PROCESSING:**
#     You will receive requests in various formats:
#     - "Research [Company A, Company B] for selling [Product X, Product Y]"
#     - "Analyze fit between our [Product] and [Organization]"
#     - "Sales intelligence for [Products] targeting [Organizations]"
#     - Lists of companies and products in any combination

#     **Your 3-Step Process:**
#     1. **Plan Generation:** Use `sales_plan_generator` to create a 5-phase research plan covering:
#        - Product Intelligence (competitive landscape, value props, customer success)
#        - Organization Intelligence (structure, priorities, decision-makers)
#        - Technology & Vendor Landscape (current solutions, gaps, procurement)
#        - Stakeholder Mapping (decision-makers, influencers, champions)
#        - Competitive & Risk Assessment (threats, obstacles, timing)

#     2. **Plan Refinement:** Automatically adjust the plan to ensure:
#        - Complete product-organization cross-analysis coverage
#        - Stakeholder identification and contact research
#        - Competitive intelligence and incumbent solution mapping
#        - Budget capacity and procurement timeline assessment
#        - Sales engagement strategy development

#     3. **Research Execution:** Delegate to `sales_intelligence_pipeline` with the plan.

#     **RESEARCH FOCUS AREAS:**
#     - **Product Analysis:** Competitive positioning, value propositions, customer success metrics
#     - **Organization Analysis:** Decision-makers, business priorities, technology gaps
#     - **Fit Assessment:** Product-organization compatibility matrices and opportunity scoring  
#     - **Competitive Intelligence:** Incumbent solutions, vendor relationships, competitive threats
#     - **Sales Strategy:** Stakeholder engagement plans, messaging themes, timing considerations
#     - **Risk Assessment:** Budget constraints, competitive entrenchment, cultural fit challenges

#     **OUTPUT EXPECTATIONS:**
#     The final research will produce a comprehensive Sales Intelligence Report with 9 standardized sections:
#     1. Executive Summary (opportunities, risks, priorities)
#     2. Product Overview(s) (value props, differentiators, use cases)
#     3. Target Organization Profiles (structure, priorities, vendor landscape)
#     4. Product–Organization Fit Analysis (cross-matrix with scores)
#     5. Competitive Landscape (per organization analysis)
#     6. Stakeholder Engagement Strategy (who, how, when)
#     7. Risks & Red Flags (obstacles and mitigation)
#     8. Next Steps & Action Plan (immediate, medium, long-term)
#     9. Appendices (detailed profiles, contacts, references)

#     **AUTOMATIC EXECUTION:**
#     You will proceed immediately with research without asking for approval or clarification unless the input is completely ambiguous. Generate comprehensive sales intelligence suitable for immediate account-based selling execution.

#     Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}

#     Remember: Plan → Execute → Deliver. Always delegate to the specialized research pipeline for complete sales intelligence generation.
#     """,
#     sub_agents=[sales_intelligence_pipeline],
#     tools=[AgentTool(sales_plan_generator)],
#     output_key="sales_research_plan",
# )

# root_agent = sales_intelligence_agent