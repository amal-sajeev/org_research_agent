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
    """Model for evaluating sales intelligence research quality with graduated standards."""
    grade: Literal["comprehensive", "sales_ready", "needs_improvement", "insufficient"] = Field(
        description="Graduated evaluation: 'comprehensive' (Tier 3), 'sales_ready' (Tier 2), 'needs_improvement' (Tier 1), 'insufficient' (requires more research)"
    )
    tier_achieved: int = Field(
        description="Quality tier achieved: 1 (minimum viable), 2 (sales ready), 3 (comprehensive)"
    )
    iteration_count: int = Field(
        description="Current iteration number in the research loop"
    )
    comment: str = Field(
        description="Detailed evaluation focusing on what's complete and what gaps remain for sales execution."
    )
    missing_critical_elements: List[str] = Field(
        default=[],
        description="List of critical missing elements for sales execution"
    )
    follow_up_queries: List[SalesResearchQuery] | None = Field(
        default=None,
        description="Maximum 4 focused follow-up searches targeting highest-impact gaps."
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
    """Enhanced escalation checker with circuit breaker logic and better state management."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        
        # Track iteration count for circuit breaker
        iteration_count = ctx.session.state.get("loop_iteration_count", 0)
        ctx.session.state["loop_iteration_count"] = iteration_count + 1
        
        # Circuit breaker: force escalation after max iterations
        if iteration_count >= config.max_search_iterations:
            logging.info(f"[{self.name}] Maximum iterations ({config.max_search_iterations}) reached. Forcing escalation.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
            return

        # Get evaluation result with enhanced validation
        evaluation_result = ctx.session.state.get("sales_research_evaluation")
        
        if not evaluation_result:
            logging.warning(f"[{self.name}] No evaluation result found in state. Keys available: {list(ctx.session.state.keys())}")
            yield Event(author=self.name, actions=EventActions(escalate=True))
            return
            
        grade = evaluation_result.get("grade")
        tier_achieved = evaluation_result.get("tier_achieved", 0)
        
        # Enhanced escalation logic with graduated standards
        if grade in ["comprehensive", "sales_ready"]:
            logging.info(f"[{self.name}] Research achieved {grade} quality (Tier {tier_achieved}). Escalating to complete report.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
            return
        elif grade == "needs_improvement":
            # Allow one more iteration for Tier 1, then escalate
            if iteration_count >= 2:
                logging.info(f"[{self.name}] Tier 1 quality achieved after {iteration_count} iterations. Escalating with current research.")
                yield Event(author=self.name, actions=EventActions(escalate=True))
                return
            else:
                logging.info(f"[{self.name}] Tier 1 quality - allowing one more improvement iteration.")
                yield Event(author=self.name)
                return
        else:  # insufficient
            logging.info(f"[{self.name}] Research insufficient (iteration {iteration_count}). Continuing loop.")
            yield Event(author=self.name)
            return


# Enhanced Structured Output Models
class StakeholderProfile(BaseModel):
    """Detailed stakeholder profile for sales targeting."""
    name: str = Field(description="Full name of stakeholder")
    title: str = Field(description="Job title/role")
    department: str = Field(description="Department or division")
    linkedin_url: str = Field(default="", description="LinkedIn profile URL if found")
    email: str = Field(default="", description="Email address if discoverable")
    phone: str = Field(default="", description="Phone number if available")
    influence_level: Literal["decision_maker", "influencer", "champion", "gatekeeper"] = Field(
        description="Level of influence in buying process"
    )
    background: str = Field(default="", description="Professional background relevant to sales approach")
    recent_activity: str = Field(default="", description="Recent posts, initiatives, or public activity")


class TechStackDetail(BaseModel):
    """Detailed technology stack information."""
    solution_category: str = Field(description="Category of technology solution")
    vendor_name: str = Field(description="Name of current vendor/provider")
    product_name: str = Field(description="Specific product or service name")
    implementation_date: str = Field(default="", description="When solution was implemented")
    contract_details: str = Field(default="", description="Contract length, renewal date if known")
    satisfaction_indicators: str = Field(default="", description="User satisfaction signals from reviews/forums")
    integration_complexity: str = Field(default="", description="Assessment of integration complexity")


class CompetitiveIntelligence(BaseModel):
    """Competitive analysis for specific product-organization combination."""
    competitor_name: str = Field(description="Name of competing solution")
    market_position: str = Field(description="Market position vs our product")
    pricing_comparison: str = Field(default="", description="Pricing comparison if available")
    feature_comparison: str = Field(default="", description="Key feature differences")
    customer_sentiment: str = Field(default="", description="Customer reviews and sentiment")
    weakness_opportunities: List[str] = Field(default=[], description="Competitive weaknesses to exploit")


# --- UTILITY AGENTS --- #
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

enhanced_sales_search = LlmAgent(
    model = config.search_model,
    name="enhanced_sales_search",
    description="Executes focused follow-up searches targeting highest-impact sales intelligence gaps.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialist sales intelligence researcher executing precision follow-up research to address the highest-impact gaps identified by evaluation.

    **CRITICAL EXECUTION REQUIREMENTS:**
    1. **Execute ALL Follow-Up Queries:** Run every single query in 'follow_up_queries' from the evaluation
    2. **Gap-Focused Approach:** Target specific missing elements rather than broad research
    3. **Efficient Integration:** Merge new findings with existing research without duplication
    4. **Quality Enhancement:** Focus on filling the most critical sales intelligence gaps

    **EXECUTION PROCESS:**
    1. **Load Context:** Review 'sales_research_findings' and 'sales_research_evaluation'
    2. **Execute Targeted Searches:** Run ALL queries in 'follow_up_queries' systematically
    3. **Prioritize High-Impact Information:**
       - Stakeholder names and contact details (highest priority)
       - Competitive intelligence and positioning gaps
       - Business priorities and strategic initiatives
       - Current vendor/technology landscape details
    4. **Integrate Seamlessly:** Combine with existing findings in organized format

    **SEARCH OPTIMIZATION STRATEGIES:**
    - Use exact query terms from follow-up list - they target specific gaps
    - When seeking stakeholders, try multiple search patterns:
      * "[Org name] [title] LinkedIn profile contact"
      * "[Org name] leadership team [department]"  
      * "[Org name] [product category] decision makers"
    - For competitive intelligence, search both directions:
      * "[Product] competitors vs [specific competitor]"
      * "[Org name] [product category] vendor selection"
    - Verify current information with "2024" or "current" qualifiers

    **INTEGRATION STANDARDS:**
    - **Preserve Unique Previous Findings:** Keep all valuable existing research
    - **Add New Intelligence:** Clearly integrate new discoveries
    - **Resolve Conflicts:** Where new info contradicts old, prioritize newer/more specific data
    - **Enhance Cross-Analysis:** Strengthen product-organization fit assessments
    - **Maintain Organization:** Use clear section headers for easy navigation

    **CRITICAL SUCCESS METRICS:**
    Your enhanced research should achieve:
    - Specific stakeholder names and roles for each organization
    - Clear competitive positioning for each product
    - Business priorities and pain points for each organization
    - Technology/vendor landscape details for each organization
    - Enhanced product-organization fit analysis with supporting evidence

    **OUTPUT FORMAT:**
    Create a comprehensive, integrated research report:

    ## Enhanced Stakeholder Intelligence
    [All decision-makers, influencers, contacts by organization]
    
    ## Comprehensive Business Intelligence
    [Strategic priorities, pain points, initiatives by organization]
    
    ## Complete Product Competitive Analysis
    [Market positioning, competitors, differentiation by product]
    
    ## Full Technology & Vendor Landscape
    [Current solutions, vendor relationships, gaps by organization]
    
    ## Enhanced Product-Organization Fit Assessment
    [Detailed cross-analysis with supporting evidence and opportunity scoring]

    **MANDATORY:** Execute every follow-up query and provide enhanced, integrated intelligence that addresses all evaluation gaps.
    """,
    tools=[google_search],
    output_key="sales_research_findings", 
    after_agent_callback=collect_research_sources_callback,
)


# --- SPECIALIZED AGENTS --- # 

# Specialized Stakeholder Discovery Agent
stakeholder_discovery_agent = LlmAgent(
    model=config.search_model,
    name="stakeholder_discovery_agent",
    description="Specialized agent focused exclusively on discovering specific stakeholder names and contact information.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialist in stakeholder discovery and contact intelligence for B2B sales research.

    **MISSION:** Find SPECIFIC NAMES of decision-makers, influencers, and gatekeepers at target organizations.

    **STAKEHOLDER DISCOVERY METHODOLOGY:**

    **Primary Search Patterns (Execute ALL for each organization):**
    
    1. **Executive Leadership Discovery:**
    - "[Org name] CEO CTO CIO CISO VP technology leadership names"
    - "[Org name] executive team management directory"
    - "[Org name] leadership team LinkedIn profiles"
    - "site:linkedin.com [Org name] CTO" (and repeat for CIO, CISO, VP Engineering)

    2. **Department Head Identification:**
    - "[Org name] IT director technology manager infrastructure names"
    - "[Org name] engineering leadership software development managers"
    - "[Org name] digital transformation officer innovation director"
    - "[Org name] procurement director purchasing manager vendor relations"

    3. **Recent Personnel Intelligence:**
    - "[Org name] recent hires LinkedIn technology management 2024"
    - "[Org name] new leadership appointments technology team"
    - "[Org name] press releases executive hires leadership changes"
    - "[Org name] company announcements new CTO CIO appointments"

    4. **Specialized Role Discovery:**
    - "[Org name] [product category] manager director specialist"
    - "[Org name] vendor management technology partnerships lead"
    - "[Org name] business development technology solutions"
    - "[Org name] innovation lab digital transformation team lead"

    **CONTACT INFORMATION STRATEGIES:**
    Once names are identified, search for:
    - "[Person name] [Org name] LinkedIn profile contact information"
    - "[Person name] [Org name] email contact speaker bio"
    - "[Org name] employee directory contact [department]"
    - "[Person name] conference speaker bio contact [industry event]"

    **VERIFICATION AND CONTEXT BUILDING:**
    For each identified stakeholder:
    - "[Person name] [Org name] background experience projects"
    - "[Person name] LinkedIn posts recent activity technology"
    - "[Person name] [Org name] technology initiatives projects"
    - "[Person name] [Org name] conference presentations speaking engagements"

    **MANDATORY SEARCH COMPLETENESS:**
    For EACH target organization, you must execute:
    - At least 8 stakeholder discovery searches targeting different roles
    - At least 3 contact information searches for identified names
    - At least 2 verification searches for key decision-makers
    - At least 1 recent activity search for each named stakeholder

    **OUTPUT REQUIREMENTS:**
    Deliver comprehensive stakeholder profiles:

    ## Stakeholder Discovery Results

    ### [Organization Name]
    **Executive Decision Makers:**
    - **[Full Name]**, [Exact Title] - [Department]
      - LinkedIn: [URL if found] | Email: [if discovered] | Phone: [if available]
      - Background: [Relevant experience, tenure, previous roles]
      - Recent Activity: [LinkedIn posts, company initiatives, speaking events]
      - Influence Assessment: [Why they matter for our product category]

    **Department Influencers:**
    - **[Full Name]**, [Exact Title] - [Department]
      - [Same detail structure as above]

    **Potential Champions:**
    - **[Full Name]**, [Exact Title] - [Department]
      - [Same detail structure plus why they'd champion our solution]

    **Procurement/Gatekeeper Contacts:**
    - **[Full Name]**, [Exact Title] - [Process they control]
      - [Contact details and process insights]

    ### Contact Intelligence Summary
    - **Immediate Reachable:** [Names with direct contact info]
    - **LinkedIn Accessible:** [Names with LinkedIn profiles for connection]
    - **Referral Required:** [Names requiring introduction or referral]
    - **Engagement Opportunities:** [Conferences, events, mutual connections]

    **CRITICAL SUCCESS METRICS:**
    - Minimum 3 specific names per organization (not just titles)
    - At least 1 decision-maker name per organization with background
    - Contact pathway identified for each named stakeholder
    - Recent activity context for engagement planning

    **SEARCH PERSISTENCE:** 
    If initial searches don't yield names, try alternative approaches:
    - Company press releases with quoted executives
    - Industry publication interviews or quotes
    - Conference speaker lists and panel participants
    - Partnership announcements with named contacts
    - Acquisition or funding announcements with leadership quotes

    Do not conclude until you have identified specific stakeholder names. Generic titles without names indicate insufficient research depth.
    """,
    tools=[google_search],
    output_key="stakeholder_intelligence",
    after_agent_callback=collect_research_sources_callback,
)

# Competitive Intelligence Specialist
competitive_intelligence_agent = LlmAgent(
    model=config.search_model,
    name="competitive_intelligence_agent", 
    description="Specialist focused on detailed competitive analysis and incumbent solution mapping.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a competitive intelligence specialist focusing on incumbent solution analysis and market positioning.

    **COMPETITIVE RESEARCH METHODOLOGY:**

    **Phase 1: Current Solution Discovery**
    For each organization, identify existing solutions:
    - "[Org name] current [product category] vendor solution provider"
    - "[Org name] technology stack [product category] tools software"
    - "[Org name] vendor partnerships technology contracts"
    - "[Org name] annual report technology investments vendor spend"

    **Phase 2: Competitor Analysis**
    For each identified competitor/incumbent:
    - "[Competitor name] vs [our product] comparison features pricing"
    - "[Competitor name] customer reviews complaints weaknesses"
    - "[Competitor name] [Org name] implementation case study"
    - "[Competitor name] market share positioning [product category]"

    **Phase 3: Market Intelligence**
    - "[Product category] market analysis leaders pricing 2024"
    - "[Org name] vendor selection criteria RFP requirements"
    - "[Product category] switching costs integration complexity"
    - "[Org name] vendor dissatisfaction contract renewal issues"

    **Phase 4: Opportunity Assessment**
    - "[Org name] technology modernization upgrade plans 2024"
    - "[Org name] contract renewal dates vendor agreements"
    - "[Org name] budget constraints technology spending limits"
    - "[Org name] [product category] procurement timeline vendor evaluation"

    **OUTPUT STRUCTURE:**

    ## Competitive Landscape Analysis

    ### [Organization Name] - Current Solutions
    **[Product Category] Incumbent:**
    - **Vendor:** [Specific vendor name]
    - **Solution:** [Specific product/service name]
    - **Implementation:** [When deployed, scope, users]
    - **Contract Status:** [Length, renewal date if known]
    - **Satisfaction Signals:** [Review scores, complaint indicators]

    ### Competitive Threat Assessment
    **Direct Competitors:**
    - **[Competitor Name]** vs Our Product
      - Market Position: [Leader/Challenger/Niche]
      - Pricing: [How they compare to our pricing]
      - Key Advantages: [Their strongest features]
      - Weaknesses: [Gaps our product addresses]
      - Customer Sentiment: [Review themes, satisfaction levels]

    ### Switching Opportunity Analysis
    **[Organization + Product Combination]:**
    - **Switching Likelihood:** [High/Medium/Low with reasoning]
    - **Key Obstacles:** [Technical, contractual, cultural barriers]
    - **Competitive Advantages:** [How we win vs incumbent]
    - **Timing Indicators:** [Contract renewal, dissatisfaction signals]
    - **Budget Assessment:** [Technology spending capacity, procurement cycles]

    Execute comprehensive competitive research for every product-organization combination. Focus on actionable intelligence that informs sales strategy and positioning.
    """,
    tools=[google_search],
    output_key="competitive_intelligence",
    after_agent_callback=collect_research_sources_callback,
)

# --- PRIMARY AGENTS --- #

# Enhanced Sales Researcher with Multi-Phase Deep Dive
enhanced_sales_researcher = LlmAgent(
    model=config.search_model,
    name="enhanced_sales_researcher",
    description="Deep-dive sales intelligence researcher with specialized search strategies for stakeholder identification and competitive analysis.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are an elite sales intelligence researcher specializing in deep stakeholder discovery and competitive analysis.

    **ENHANCED RESEARCH METHODOLOGY:**
    Execute systematic multi-phase research with escalating search sophistication.

    **PHASE 1: FOUNDATION RESEARCH (Standard Searches)**
    Execute baseline searches for each entity:

    For Products:
    - "[Product name] competitors comparison 2024"
    - "[Product name] pricing model features benefits"
    - "[Product name] customer reviews case studies ROI"
    - "[Product category] market analysis leaders"

    For Organizations:
    - "[Org name] company information revenue size employees"
    - "[Org name] news 2024 strategic initiatives announcements"
    - "[Org name] leadership team executives organizational chart"
    - "[Org name] technology stack vendor partnerships"

    **PHASE 2: STAKEHOLDER DEEP DIVE (Advanced People Search)**
    Critical: Use multiple search patterns to find SPECIFIC NAMES:

    Executive Leadership Discovery:
    - "[Org name] CEO CTO CIO CISO VP leadership team LinkedIn"
    - "[Org name] executive team management board directors"
    - "[Org name] leadership changes 2024 new hires executives"
    - "site:linkedin.com [Org name] CTO OR CIO OR 'VP Technology'"

    Department-Specific Leaders:
    - "[Org name] IT director technology manager infrastructure"
    - "[Org name] procurement director purchasing manager vendor relations"
    - "[Org name] digital transformation officer innovation leader"
    - "[Org name] [product category] manager director lead"

    Recent Personnel Intelligence:
    - "[Org name] recent hires LinkedIn technology [product category]"
    - "[Org name] job postings technology roles hiring"
    - "[Org name] conference speakers technology presentations 2024"
    - "[Org name] press releases leadership appointments"

    **PHASE 3: COMPETITIVE & VENDOR INTELLIGENCE (Deep Market Analysis)**
    
    Current Solution Mapping:
    - "[Org name] current [product category] solution vendor"
    - "[Org name] technology vendors partnerships contracts"
    - "[Org name] [specific competitor] implementation usage"
    - "[Org name] vendor selection RFP requirements [product category]"

    Competitive Threat Assessment:
    - "[Main competitor] [Org name] customer reference case study"
    - "[Org name] [product category] budget spending technology investment"
    - "[Org name] vendor dissatisfaction issues problems reviews"
    - "[Org name] technology modernization upgrade plans"

    **PHASE 4: BUSINESS INTELLIGENCE (Strategic Context)**
    
    Business Priority Mapping:
    - "[Org name] business strategy 2024 priorities initiatives"
    - "[Org name] challenges pain points business problems"
    - "[Org name] growth plans expansion investment areas"
    - "[Org name] quarterly earnings technology spending"

    Buying Signal Detection:
    - "[Org name] RFP tender procurement [product category] 2024"
    - "[Org name] budget allocation technology investment"
    - "[Org name] vendor evaluation selection process"
    - "[Org name] technology roadmap digital transformation"

    **PHASE 5: CONTACT & ENGAGEMENT INTELLIGENCE**
    
    Contact Discovery:
    - "[Person name] [Org name] contact email phone LinkedIn"
    - "[Org name] sales contact business development"
    - "[Org name] technology evaluation team contact"
    - "[Org name] procurement process vendor onboarding contact"

    Engagement Context:
    - "[Person name] [Org name] recent posts LinkedIn activity"
    - "[Org name] events conferences technology presentations"
    - "[Org name] partnership announcements vendor relationships"
    - "[Org name] industry associations technology groups"

    **ENHANCED SEARCH TACTICS:**
    - **Name Discovery:** Use multiple search angles: company + title, department + company, recent hires + company
    - **Verification:** Cross-reference names across LinkedIn, company websites, and press releases
    - **Context Building:** Search for each identified person individually for background and contact info
    - **Relationship Mapping:** Search for connections between stakeholders and current vendors
    - **Intelligence Validation:** Verify competitive intelligence with multiple sources

    **CRITICAL SUCCESS METRICS:**
    Your research MUST deliver:
    - At least 3 specific stakeholder names per organization with titles and departments
    - Current technology vendors and contract details for each organization
    - Specific competitive threats and incumbent solutions per product-organization pair
    - Business priorities and strategic initiatives for each organization
    - Contact information and engagement context for key stakeholders

    **OUTPUT REQUIREMENT:**
    Structure your findings with clear sections:
    
    ## Stakeholder Intelligence (Names Required)
    ### [Organization Name]
    - **Decision Makers:** [Name], [Title], [Department] - [Background/Context]
    - **Influencers:** [Name], [Title], [Department] - [Recent Activity/Posts]
    - **Champions:** [Name], [Title], [Department] - [Why They'd Support Solution]
    - **Gatekeepers:** [Name], [Title] - [Contact Method/Process]

    ## Technology & Vendor Landscape (Current Solutions)
    ### [Organization Name]
    - **[Product Category]:** Current vendor, solution name, implementation date, contract status
    - **Integration Partners:** Key technology partnerships affecting our product fit
    - **Vendor Satisfaction:** Review signals, complaint indicators, renewal likelihood

    ## Competitive Intelligence (Specific Threats)
    ### [Product] vs [Organization] 
    - **Incumbent Solutions:** [Specific competitor], [Market position], [Customer satisfaction]
    - **Competitive Weaknesses:** [Specific gaps our product addresses]
    - **Switching Barriers:** [Technical, contractual, cultural obstacles]

    ## Business Context & Buying Signals
    ### [Organization Name]
    - **Strategic Priorities:** [Specific 2024 initiatives]
    - **Pain Points:** [Documented business challenges]
    - **Technology Investment:** [Budget indicators, spending patterns]
    - **Procurement Signals:** [RFPs, vendor evaluations, hiring patterns]

    Execute ALL phases systematically. Do not conclude until you have specific stakeholder names and competitive intelligence for every product-organization combination.
    """,
    tools=[google_search],
    output_key="enhanced_sales_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

# Enhanced Stakeholder-Focused Evaluator
enhanced_sales_evaluator = LlmAgent(
    model=config.critic_model,
    name="enhanced_sales_evaluator", 
    description="Evaluates sales intelligence with strict focus on stakeholder identification and actionable sales insights.",
    instruction=f"""
    You are a senior sales intelligence quality assessor with expertise in account-based selling requirements.

    **ENHANCED EVALUATION FRAMEWORK:**

    **TIER 3 - COMPREHENSIVE (grade: "comprehensive"):**
    STAKEHOLDER REQUIREMENTS:
    - Minimum 4 specific stakeholder names per organization with titles and departments
    - At least 2 decision-makers identified with background context
    - Contact information (LinkedIn, email, or phone) for at least 50% of identified stakeholders
    - Recent activity or engagement context for key stakeholders

    COMPETITIVE INTELLIGENCE:
    - Current incumbent solutions identified for each product category per organization
    - Specific competitive positioning analysis with feature/price comparisons
    - Vendor satisfaction indicators and contract renewal timelines
    - Competitive weaknesses and switching opportunity assessment

    BUSINESS CONTEXT:
    - Documented strategic priorities and pain points for each organization
    - Technology investment patterns and budget indicators
    - Procurement processes and vendor selection criteria
    - Buying signals and timing indicators

    **TIER 2 - SALES READY (grade: "sales_ready"):**
    STAKEHOLDER REQUIREMENTS:
    - Minimum 2 specific stakeholder names per organization with accurate titles
    - At least 1 confirmed decision-maker per organization
    - Department or reporting structure context for each stakeholder
    - LinkedIn profiles or company directory confirmation for key contacts

    COMPETITIVE INTELLIGENCE:
    - At least 1 current vendor/solution identified per organization in relevant category
    - Basic competitive positioning for each product vs. incumbents
    - General market position and customer sentiment for main competitors
    - Identified switching barriers or competitive advantages

    BUSINESS CONTEXT:
    - Core business priorities or strategic initiatives identified per organization
    - Basic technology modernization needs or digital transformation indicators
    - General budget capacity or procurement approach signals

    **TIER 1 - NEEDS IMPROVEMENT (grade: "needs_improvement"):**
    MINIMUM STAKEHOLDER REQUIREMENTS:
    - At least 1 specific stakeholder name per organization with correct title
    - Department identification for technology or procurement roles
    - Some evidence of decision-making structure or process

    MINIMUM COMPETITIVE INTELLIGENCE:
    - General competitive landscape awareness for each product
    - Some current technology vendor identification per organization
    - Basic market positioning context

    **INSUFFICIENT (grade: "insufficient"):**
    - Generic titles without specific names ("CTO", "IT Director" without actual names)
    - No current vendor/solution identification
    - No specific business priorities or pain points
    - No competitive analysis or market positioning

    **STAKEHOLDER NAME VALIDATION:**
    When evaluating stakeholder research, verify:
    - Are these actual people with real names and titles?
    - Do the titles match the organization size and structure?
    - Is there context about their role in technology decisions?
    - Are there engagement signals (recent posts, initiatives, public activity)?

    **FOLLOW-UP QUERY PRIORITIZATION:**
    When generating follow-up queries, prioritize in this order:
    1. **Missing Stakeholder Names:** "[Org name] CTO name current 2024" or "[Org name] IT director LinkedIn profile"
    2. **Incomplete Competitive Intel:** "[Org name] current [product category] vendor solution"
    3. **Missing Business Context:** "[Org name] strategic priorities technology investment 2024"
    4. **Contact Information Gaps:** "[Person name] [Org name] contact LinkedIn email"

    **EVALUATION PROCESS:**
    1. Count specific stakeholder names (not just titles)
    2. Verify competitive intelligence completeness
    3. Assess business context depth
    4. Check cross-analysis coverage
    5. Generate targeted follow-up queries for highest-impact gaps

    **STRICT NAMING REQUIREMENT:**
    Research that provides only generic titles ("The CTO", "IT Director") without actual names automatically qualifies as "needs_improvement" or "insufficient" regardless of other content quality.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Your response must be a single, raw JSON object validating against the 'SalesFeedback' schema.
    """,
    output_schema=SalesFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="sales_research_evaluation",
)

# Updated Sales Plan Generator with Enhanced Requirements
sales_plan_generator = LlmAgent(
    model=config.search_model,
    name="sales_plan_generator",
    description="Generates comprehensive sales intelligence research plans with mandatory stakeholder discovery and competitive analysis requirements.",
    instruction=f"""
    You are an expert sales intelligence strategist specializing in account-based selling research with mandatory stakeholder identification and competitive analysis requirements.

    **CRITICAL ENHANCEMENT REQUIREMENTS:**
    Your research plan must GUARANTEE discovery of:
    1. **Specific stakeholder names** (not just titles) for each organization
    2. **Current incumbent solutions** and vendor relationships
    3. **Detailed competitive intelligence** for positioning
    4. **Contact information** and engagement pathways
    5. **Business context** for sales conversation relevance

    **MANDATORY RESEARCH PLAN COMPONENTS:**

    **Phase 1: Enhanced Product Intelligence**
    For each product:
    - Deep competitive landscape mapping with specific competitor analysis
    - Customer testimonial and case study collection with ROI metrics
    - Pricing strategy and value proposition documentation
    - Technical integration capabilities and requirement analysis
    - Market position assessment vs. identified competitors

    **Phase 2: Deep Organization Intelligence** 
    For each target organization:
    - **Leadership Discovery:** Specific names of CTO, CIO, VP Technology, IT Directors
    - **Department Mapping:** Technology team structure with named personnel
    - **Decision Process:** Vendor selection criteria and procurement workflows  
    - **Financial Intelligence:** Technology budget, spending patterns, contract cycles
    - **Strategic Context:** Business priorities, digital transformation initiatives

    **Phase 3: Stakeholder Identification & Mapping** (NEW MANDATORY PHASE)
    For each target organization:
    - **Executive Stakeholder Discovery:** Search patterns for finding named executives
    - **Department Leader Research:** Specific search strategies for IT/Technology leadership
    - **Recent Hire Analysis:** New leadership in technology roles
    - **Contact Intelligence:** LinkedIn, company directory, conference speaker searches
    - **Engagement Context:** Recent posts, initiatives, speaking engagements

    **Phase 4: Incumbent Solution & Vendor Analysis** (ENHANCED)
    For each target organization:
    - **Current Vendor Mapping:** Specific technology providers and solutions
    - **Contract Intelligence:** Renewal dates, satisfaction levels, switching signals
    - **Integration Analysis:** Technology stack compatibility and gaps
    - **Vendor Relationship Assessment:** Partnership satisfaction and renewal likelihood

    **Phase 5: Cross-Analysis & Opportunity Assessment**
    - **Product-Organization Fit Matrix:** Detailed compatibility scoring
    - **Competitive Positioning:** How each product competes vs. incumbents per organization
    - **Stakeholder Influence Mapping:** Decision-maker vs. influencer assessment
    - **Sales Strategy Framework:** Engagement sequencing and messaging themes

    **SEARCH STRATEGY SPECIFICATIONS:**

    **Stakeholder Discovery Searches (Mandatory for each org):**
    - "[Org name] CTO name current 2024 LinkedIn"
    - "[Org name] CIO technology leadership team names"
    - "[Org name] VP engineering software development director"
    - "[Org name] IT director infrastructure manager names"
    - "[Org name] digital transformation officer innovation lead"
    - "[Org name] recent hires technology leadership 2024"
    - "site:linkedin.com [Org name] technology leadership"

    **Incumbent Solution Discovery (Mandatory for each org):**
    - "[Org name] current [product category] vendor solution"
    - "[Org name] technology stack software tools vendors"
    - "[Org name] vendor partnerships technology contracts"
    - "[Org name] [specific competitor] implementation usage"

    **Competitive Intelligence (Mandatory for each product):**
    - "[Product name] vs [identified competitor] comparison"
    - "[Product name] competitive analysis market position"
    - "[Product category] market leaders customer satisfaction"
    - "[Product name] customer reviews weaknesses complaints"

    **Contact & Engagement Intelligence:**
    - "[Identified person name] [Org name] contact LinkedIn email"
    - "[Org name] procurement contact vendor onboarding process"
    - "[Org name] technology evaluation team contact information"

    **SUCCESS CRITERIA FOR PLAN:**
    The plan must enable discovery of:
    - Minimum 3 specific stakeholder names per organization
    - Current technology vendors and solutions per organization
    - Competitive positioning analysis for each product
    - Contact pathways for identified stakeholders  
    - Business priorities and pain points for sales conversation

    **QUALITY ASSURANCE REQUIREMENTS:**
    - Search patterns designed to find NAMES, not just titles
    - Multiple search approaches for stakeholder discovery
    - Verification searches for competitive intelligence
    - Contact discovery strategies for identified personnel
    - Business context searches for sales relevance

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    
    Create a research plan that GUARANTEES specific stakeholder names and detailed competitive intelligence for actionable account-based selling.
    """,
    output_key="enhanced_sales_research_plan",
    tools=[google_search],
)

# Enhanced Report Composer with Stakeholder Focus
enhanced_sales_report_composer = LlmAgent(
    model=config.critic_model,
    name="enhanced_sales_report_composer",
    include_contents="none",
    description="Composes complete sales intelligence reports using only research data, with graceful handling of missing information.",
    instruction="""
    You are an expert sales intelligence report writer who creates complete, professional reports using ONLY the research data provided.

    **CORE PRINCIPLE:** Generate a complete report using available research findings. Where specific data is missing, use professional language that acknowledges gaps without breaking report flow.

    ---
    ### INPUT DATA SOURCES
    * Research Plan: `{enhanced_sales_research_plan}`
    * Foundation Research: `{enhanced_sales_research_findings}` 
    * Stakeholder Intelligence: `{stakeholder_intelligence}`
    * Competitive Intelligence: `{competitive_intelligence}`
    * Citation Sources: `{sources}`
    * Report Structure: `{sales_report_sections}`

    ---
    ### REPORT COMPOSITION APPROACH

    **1. Use Available Data First:**
    - Extract all stakeholder names, titles, and contact info from research
    - Include all competitive intelligence and current solutions identified
    - Document all business priorities and strategic initiatives found
    - Present all product-organization fit analysis from research

    **2. Handle Missing Information Professionally:**
    Instead of asking for data, use phrases like:
    - "Additional stakeholder identification in progress"
    - "Further competitive analysis recommended"
    - "Contact information to be verified through LinkedIn outreach"
    - "Procurement process details require direct organizational contact"

    **3. Complete Report Structure:**
    Always generate all 9 sections, populating with available research:

    # 1. Executive Summary
    - Summarize research scope and key findings
    - Highlight discovered opportunities and stakeholders
    - Note competitive landscape insights
    - Present risk assessment based on research

    # 2. Product Overview(s)
    - Use research findings about each product
    - Include competitive positioning discovered
    - Present value propositions and use cases
    - Document customer success data found

    # 3. Target Organization Profiles
    ## 3.1 Organization Summary
    - Present company data from research
    - Include recent news and initiatives found

    ## 3.2 Stakeholder Profiles & Influence Mapping
    Create tables with discovered stakeholders:

    **Executive Decision Makers:**
    | Name | Title | Department | Contact Info | Background | Next Steps |
    |------|-------|------------|-------------|------------|------------|
    | [Name if found] | [Title] | [Dept] | [LinkedIn URL/Email if found] | [Background from research] | [Engagement approach] |
    | *Additional CTO research needed* | CTO | Technology | *LinkedIn search required* | *Background verification in progress* | *Initial LinkedIn connection* |

    **Department Influencers:**
    | Name | Title | Contact Method | Recent Activity | Sales Relevance |
    |------|-------|----------------|-----------------|-----------------|
    | [Names from research] | [Titles] | [Contact info found] | [Activity noted] | [Relevance assessment] |

    ## 3.3 Business Priorities & Pain Points
    - Present strategic initiatives found in research
    - Include pain points and challenges identified
    - Note technology gaps discovered

    ## 3.4 Current Vendor & Solution Landscape
    - List current vendors/solutions identified
    - Include contract information if found
    - Note satisfaction indicators from research

    # 4. Product–Organization Fit Analysis
    Create fit analysis using research findings:

    | Product | Organization | Strategic Fit | Key Stakeholders | Competitive Risk | Opportunity Score |
    |---------|--------------|---------------|------------------|------------------|-------------------|
    | [Product] | [Org] | [Assessment from research] | [Named contacts found] | [Threats identified] | [Score with reasoning] |

    # 5. Competitive Landscape
    ## 5.1 Current Solution Landscape
    | Organization | Product Category | Current Solution | Contract Status | Opportunity |
    |--------------|------------------|------------------|-----------------|-------------|
    | [Org] | [Category] | [Solution from research] | [Status if known] | [Switching opportunity] |

    ## 5.2 Competitive Analysis
    Present competitive intelligence found in research for each product-organization pair.

    # 6. Stakeholder Engagement Strategy
    ## 6.1 Primary Target Strategy
    **[Organization]:**
    - **Primary Contact:** [Name from research or "CTO identification in progress"]
    - **Engagement Channel:** [LinkedIn/Email based on contact info found]
    - **Message Theme:** [Based on business priorities discovered]
    - **Next Action:** [Specific next step based on available data]

    ## 6.2 Recommended Engagement Sequence
    Create realistic timeline based on identified contacts and research gaps.

    # 7. Risks & Red Flags
    Document risks identified in research and note areas requiring additional investigation.

    # 8. Next Steps & Action Plan
    ## Immediate Actions (Next 7-14 days)
    - List specific actions based on research findings
    - Include stakeholder research tasks where gaps exist

    ## Medium-Term Strategy (Next 1-3 months)
    - Engagement plans for identified contacts
    - Competitive positioning activities

    # 9. Appendices
    ## A. Detailed Stakeholder Research Notes
    Include all stakeholder information found, noting gaps for follow-up.

    ## B. Competitive Intelligence Summary
    Present detailed competitive findings.

    ## C. Research Methodology Notes
    Document search strategies used and recommend additional research approaches.

    **CRITICAL INSTRUCTIONS:**
    1. **Generate complete report** regardless of data gaps
    2. **Use professional language** for missing information ("research in progress" not "I don't have this")
    3. **Create actionable sections** even with incomplete data
    4. **Include all research findings** in appropriate sections
    5. **Maintain professional tone** throughout
    6. **Use proper citations** with `<cite source="src-ID" />` format
    7. **Never ask user for additional input** - work with provided research data

    **MISSING DATA HANDLING:**
    - Stakeholders: "Additional leadership identification recommended via LinkedIn research"
    - Competitive: "Further competitive analysis of [specific gaps] in development"  
    - Contact Info: "Contact verification through professional networks in progress"
    - Business Context: "Additional strategic priority research through company communications planned"

    Generate a complete, professional sales intelligence report that sales teams can use immediately while noting areas for continued research expansion.
    """,
    output_key="sales_intelligence_agent",
    after_agent_callback=citation_replacement_callback,
)

# Enhanced Multi-Phase Research Pipeline
sales_intelligence_pipeline = SequentialAgent(
    name="sales_intelligence_pipeline",
    description="Enhanced multi-agent pipeline for comprehensive sales intelligence with specialized stakeholder and competitive research.",
    sub_agents=[
        sales_section_planner,
        enhanced_sales_researcher,  # Enhanced foundation research
        stakeholder_discovery_agent,  # Specialized stakeholder discovery
        competitive_intelligence_agent,  # Specialized competitive analysis
        LoopAgent(
            name="enhanced_quality_assurance_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                enhanced_sales_evaluator,  # Enhanced evaluator with stricter standards
                SalesEscalationChecker(name="sales_escalation_checker"),
                enhanced_sales_search,
            ],
        ),
        enhanced_sales_report_composer  # Final report composition
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

