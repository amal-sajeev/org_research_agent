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
from google.adk.models import Gemini
from pydantic import BaseModel, Field

from ...config import config
from .target_template import TARGET_TEMPLATE

# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Model representing a specific search query for organizational research."""
    search_query: str = Field(
        description="A highly specific and targeted query for organizational web search, focusing on company information, social media, and public perception."
    )
    research_phase: str = Field(
        description="The research phase this query belongs to: 'foundation', 'market_intelligence', 'deep_dive', or 'risk_assessment'"
    )

class CompactResearchFindings(BaseModel):
    """Structured model for research findings that prevents token overflow."""
    company_basics: dict = Field(default_factory=dict, description="Core company information")
    financial_data: dict = Field(default_factory=dict, description="Financial metrics and data")
    leadership_info: dict = Field(default_factory=dict, description="Leadership and personnel data")
    market_position: dict = Field(default_factory=dict, description="Market and competitive data")
    recent_developments: list = Field(default_factory=list, description="List of recent developments with dates")
    risk_factors: list = Field(default_factory=list, description="Identified risk factors")
    sales_intelligence: dict = Field(default_factory=dict, description="Sales-relevant insights")
    source_count: int = Field(default=0, description="Number of sources used")

class ResearchSection(BaseModel):
    """Model for a single research section with content and citations."""
    section_id: str = Field(description="Unique identifier for the section")
    title: str = Field(description="Section title")
    content: str = Field(description="Detailed section content with inline citations")
    subsections: list[dict] = Field(default=[], description="List of subsections with title and content")

class Feedback(BaseModel):
    """Model for providing evaluation feedback on organizational research quality."""
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the research meets organizational intelligence standards, 'fail' if it needs more depth."
    )
    comment: str = Field(
        description="Detailed evaluation focusing on completeness of company information, source diversity, and sales-relevant insights."
    )
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description="Specific follow-up searches needed to fill organizational intelligence gaps.",
    )

# --- Enhanced Callbacks with Token Management ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources with size limits to prevent token overflow."""
    session = callback_context._invocation_context.session
    url_to_short_id = callback_context.state.get("url_to_short_id", {})
    sources = callback_context.state.get("sources", {})
    id_counter = len(url_to_short_id) + 1
    
    # Limit total sources to prevent token overflow
    MAX_SOURCES = 25
    
    for event in session.events:
        if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
            continue
        
        # Stop if we've reached max sources
        if len(sources) >= MAX_SOURCES:
            logging.warning(f"Reached maximum source limit ({MAX_SOURCES}). Skipping additional sources.")
            break
        
        chunks_info = {}
        for idx, chunk in enumerate(event.grounding_metadata.grounding_chunks):
            if not chunk.web:
                continue
            
            url = chunk.web.uri
            title = chunk.web.title if chunk.web.title != chunk.web.domain else chunk.web.domain
            domain = chunk.web.domain or "unknown"
            
            # Handle cases where title might be None
            if not title or title == domain:
                title = domain or "Unknown Source"
            
            # Truncate long titles to save tokens
            if len(title) > 100:
                title = title[:97] + "..."
            
            if url not in url_to_short_id and len(sources) < MAX_SOURCES:
                short_id = f"src-{id_counter}"
                url_to_short_id[url] = short_id
                sources[short_id] = {
                    "short_id": short_id,
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "supported_claims": [],
                    "access_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "source_type": _classify_source_type(domain, url)
                }
                id_counter += 1
                chunks_info[idx] = short_id
        
        # Limit claims per source to prevent token overflow
        MAX_CLAIMS_PER_SOURCE = 3
        
        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        short_id = chunks_info[chunk_idx]
                        if len(sources[short_id]["supported_claims"]) < MAX_CLAIMS_PER_SOURCE:
                            confidence = confidence_scores[i] if i < len(confidence_scores) else 0.5
                            text_segment = support.segment.text if support.segment else ""
                            # Truncate long text segments
                            if len(text_segment) > 200:
                                text_segment = text_segment[:197] + "..."
                            sources[short_id]["supported_claims"].append({
                                "text_segment": text_segment,
                                "confidence": confidence,
                            })
    
    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources
    
    # Log source collection stats
    logging.info(f"Collected {len(sources)} sources with total claims: {sum(len(s['supported_claims']) for s in sources.values())}")

def structured_findings_callback(callback_context: CallbackContext) -> None:
    """Converts research output to structured format to reduce token usage."""
    try:
        # Get the raw research findings
        raw_findings = callback_context.state.get("organizational_research_findings", "")
        
        # Create structured summary instead of keeping full text
        structured_summary = CompactResearchFindings(
            source_count=len(callback_context.state.get("sources", {}))
        )
        
        # Store structured version and remove raw text to save tokens
        callback_context.state["structured_research_data"] = structured_summary.dict()
        callback_context.state["research_summary_token_count"] = len(str(structured_summary))
        
        # Keep only last 2000 chars of findings to prevent overflow
        if len(raw_findings) > 2000:
            callback_context.state["organizational_research_findings"] = raw_findings[-2000:]
            logging.info(f"Truncated research findings from {len(raw_findings)} to 2000 characters")
        
    except Exception as e:
        logging.error(f"Error in structured findings callback: {e}")

def _classify_source_type(domain: str, url: str) -> str:
    """Classify source type based on domain and URL patterns."""
    # Handle None values safely
    domain_lower = (domain or "").lower()
    url_lower = (url or "").lower()
    
    if any(x in domain_lower for x in ['linkedin.com', 'twitter.com', 'facebook.com', 'instagram.com']):
        return "Social Media"
    elif any(x in domain_lower for x in ['sec.gov', 'edgar', 'bloomberg.com', 'reuters.com']):
        return "Financial"
    elif any(x in domain_lower for x in ['crunchbase.com', 'pitchbook.com']):
        return "Business Database"
    elif 'news' in domain_lower or any(x in domain_lower for x in ['cnn.com', 'bbc.com', 'wsj.com']):
        return "News Media"
    elif any(x in url_lower for x in ['about', 'company', 'leadership', 'team']):
        return "Company Official"
    else:
        return "Industry/Other"

def citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Replaces citation tags in a report with Wikipedia-style clickable numbered references."""
    final_report = callback_context.state.get("organizational_intelligence_report", "")
    sources = callback_context.state.get("sources", {})

    # Limit references to prevent token overflow
    MAX_REFERENCES = 15
    limited_sources = dict(list(sources.items())[:MAX_REFERENCES])

    # Assign each short_id a numeric index
    short_id_to_index = {}
    for idx, short_id in enumerate(sorted(limited_sources.keys()), start=1):
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
        source_info = limited_sources[short_id]
        domain = source_info.get('domain', '')
        references += (
            f"<p id=\"ref{idx}\">[{idx}] "
            f"<a href=\"{source_info['url']}\">{source_info['title']}</a>"
            f"{f' ({domain})' if domain else ''}</p>\n"
        )

    processed_report += references
    
    # Store final report and clear intermediate data to save tokens
    callback_context.state["organizational_intelligence_agent"] = processed_report
    
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])

# --- Enhanced Agent Definitions ---
organizational_plan_generator = LlmAgent(
    model=config.search_model,
    name="organizational_plan_generator",
    description="Generates focused organizational research plans with exact name matching.",
    instruction=f"""
    You are an expert organizational intelligence strategist. Create a concise, focused research plan.
    
    **MISSION:** Create a systematic research plan with EXACT name matching for efficient investigation.

    **CRITICAL REQUIREMENTS:**
    - Always use the COMPLETE, EXACT organization name in quotation marks
    - Generate 8-12 focused search queries maximum (not 20+)
    - Prioritize high-impact searches that yield maximum intelligence
    - Focus on recent information (last 12-18 months)

    **FOCUSED RESEARCH AREAS (Prioritized):**

    **Foundation (3-4 searches):**
    - "\"[EXACT Company Name]\" official website about company business model"
    - "\"[EXACT Company Name]\" leadership executives CEO team LinkedIn"
    - "\"[EXACT Company Name]\" revenue financial performance employees size"

    **Intelligence Gathering (3-4 searches):**
    - "\"[EXACT Company Name]\" news recent developments 2024 2025"
    - "\"[EXACT Company Name]\" competitors market position industry"
    - "\"[EXACT Company Name]\" funding investors partnerships acquisitions"

    **Assessment (2-4 searches):**
    - "\"[EXACT Company Name]\" technology stack digital transformation"
    - "\"[EXACT Company Name]\" customer reviews case studies testimonials"
    - "\"[EXACT Company Name]\" risks challenges controversies" (if needed)

    **OUTPUT:** Generate concise, actionable search plan focusing on essential business intelligence.
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    output_key="research_plan",
    tools=[google_search],
)

organizational_section_planner = LlmAgent(
    model=config.worker_model,
    name="organizational_section_planner",
    description="Creates efficient report structure for organizational research.",
    instruction="""
    Create a focused markdown outline with these essential sections:

    # 1. Executive Summary
    # 2. Company Foundation  
    # 3. Financial Intelligence
    # 4. Leadership Analysis
    # 5. Market Position
    # 6. Recent Developments
    # 7. Risk Assessment
    # 8. Sales Intelligence
    # 9. Recommendations

    Keep section descriptions concise - the focus is on efficient structure, not detailed instructions.
    """,
    output_key="report_sections",
)

organizational_researcher = LlmAgent(
    model=config.search_model,
    name="organizational_researcher",
    description="Focused organizational researcher with token-efficient output.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a business intelligence researcher focused on efficient, comprehensive company analysis.

    **CORE PROTOCOL:**
    1. Execute searches systematically using EXACT company name in quotes
    2. Collect key information across all business areas
    3. **CRITICAL:** Provide STRUCTURED, CONCISE findings - not lengthy narratives

    **RESEARCH EXECUTION:**
    - Use complete, exact organization name in quotation marks for all searches
    - Focus on authoritative sources (company website, LinkedIn, financial databases)
    - Prioritize recent information (12-18 months)
    - Balance positive and negative findings

    **OUTPUT FORMAT - STRUCTURED KEY FINDINGS:**
    Organize findings as structured data points, not lengthy paragraphs:

    **Company Basics:**
    - Legal name, industry, founded, headquarters
    - Business model, core products/services
    - Employee count, company size
    - Geographic presence

    **Financial Intelligence:**
    - Revenue figures (latest available)
    - Funding history and investors
    - Financial health indicators
    - Growth metrics and trends

    **Leadership Profile:**
    - CEO and key executives (names, backgrounds)
    - Recent leadership changes
    - Board composition
    - Decision-making structure

    **Market Analysis:**
    - Primary competitors
    - Market position and share
    - Competitive advantages
    - Industry dynamics

    **Recent Developments:**
    - Major news and announcements (with dates)
    - Strategic partnerships and deals
    - Product launches and initiatives
    - Expansion activities

    **Risk & Opportunity Factors:**
    - Business risks and challenges
    - Growth opportunities
    - Market threats
    - Buying signals for sales

    **EFFICIENCY REQUIREMENTS:**
    - Use bullet points and structured lists
    - Avoid repetitive information
    - Focus on facts and specific data points
    - Cite sources appropriately but concisely
    - Maximum 1500 words for all findings combined

    This structured approach prevents token overflow while maintaining comprehensive coverage.
    """,
    tools=[google_search],
    output_key="compact_research_data",
    after_agent_callback=collect_research_sources_callback,
)

organizational_evaluator = LlmAgent(
    model=config.critic_model,
    name="organizational_evaluator",
    description="Efficient evaluation specialist for research quality assessment.",
    instruction="""
    Evaluate research completeness against these focused criteria:

    **EVALUATION CHECKLIST (Pass requires 70% coverage):**
    - Company identity and basic information ✓
    - Business model and revenue clarity ✓
    - Leadership team identification ✓
    - Financial performance indicators ✓
    - Market position understanding ✓
    - Recent developments coverage ✓
    - Sales-relevant intelligence ✓

    **QUICK ASSESSMENT:**
    - **PASS:** Core areas covered with specific data points and multiple sources
    - **FAIL:** Major gaps in foundational information or lack of depth

    **FOLLOW-UP QUERIES (if FAIL):**
    Generate exactly 2-3 targeted searches to fill critical gaps only.

    Keep evaluation concise and actionable.
    """,
    output_schema=Feedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
)

enhanced_organizational_search = LlmAgent(
    model=config.search_model,
    name="enhanced_organizational_search",
    description="Targeted gap-filling researcher with token-efficient execution.",
    instruction="""
    Execute focused follow-up research to address specific gaps identified in evaluation.

    **EXECUTION PROTOCOL:**
    1. Review evaluation feedback for specific missing information
    2. Execute ALL queries from 'follow_up_queries' efficiently
    3. Focus on filling identified gaps only
    4. Integrate findings with existing research data

    **OUTPUT:** Provide ONLY the new information found - do not repeat existing research.
    Use structured bullet points for efficiency.

    **TOKEN EFFICIENCY:**
    - Focus on facts and data points
    - Avoid lengthy descriptions
    - Update existing research categories only
    - Maximum 800 words for gap-filling research
    """,
    tools=[google_search],
    output_key="gap_fill_research",
    after_agent_callback=structured_findings_callback,
)

organizational_report_composer = LlmAgent(
    model=config.critic_model,
    name="organizational_report_composer",
    description="Expert business intelligence report writer with efficient content synthesis.",
    instruction="""
    Transform structured research data into a professional markdown organizational intelligence report.

    **INPUT DATA:**
    - Compact research data: `{compact_research_data}`
    - Report structure: `{report_sections}`
    - Citation sources: `{sources}`
    - Gap-fill research: `{gap_fill_research}` (if available)

    **REPORT COMPOSITION:**
    1. Use the structured research data to populate each section
    2. Maintain professional tone and comprehensive coverage
    3. Include proper citations using `<cite source="src-ID" />` format
    4. Focus on actionable business intelligence

    **EFFICIENCY REQUIREMENTS:**
    - Use structured research data efficiently
    - Avoid repetitive content
    - Focus on high-value information
    - Include specific metrics and data points
    - Target 2000-2500 words for complete report

    **CITATION PROTOCOL:**
    - Cite all factual claims with `<cite source="src-ID" />`
    - Use sources from the sources dictionary
    - Place citations immediately after relevant statements

    Generate a comprehensive yet efficient organizational intelligence report.
    """,
    output_key="organizational_intelligence_report",
    after_agent_callback=citation_replacement_callback,
)

org_html_composer = LlmAgent(
    model=config.critic_model,
    name="client_org_html_composer",
    include_contents="none",
    description="Composes efficient HTML sales analysis report.",
    instruction=TARGET_TEMPLATE,
    output_key="org_html",
)

# --- Enhanced Loop Control Agent ---
class EscalationChecker(BaseAgent):
    """Efficient escalation checker with improved detection and safety controls."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Improved escalation logic with token-efficient state checking."""
        
        evaluation_result = None
        
        try:
            # Check for evaluation in session state
            evaluation_result = ctx.session.state.get("research_evaluation")
            
            # Check recent events if not found in state
            if not evaluation_result:
                for event in reversed(ctx.session.events[-5:]):  # Check last 5 events only
                    if hasattr(event, 'author') and 'evaluator' in str(event.author).lower():
                        content = str(event.content) if hasattr(event, 'content') else ""
                        if '"grade"' in content:
                            if '"pass"' in content.lower():
                                evaluation_result = {"grade": "pass"}
                                break
                            elif '"fail"' in content.lower():
                                evaluation_result = {"grade": "fail"}
                                break

        except Exception as e:
            logging.error(f"[{self.name}] Error during evaluation detection: {e}")

        # Determine escalation
        should_escalate = False
        grade_found = "unknown"
        
        if evaluation_result:
            if isinstance(evaluation_result, dict):
                grade = str(evaluation_result.get("grade", "")).lower().strip()
                grade_found = grade
                should_escalate = grade == "pass"
            elif hasattr(evaluation_result, 'grade'):
                grade = str(evaluation_result.grade).lower().strip()
                grade_found = grade
                should_escalate = grade == "pass"

        # Safety mechanism - limit iterations
        loop_counter = ctx.session.state.get("escalation_check_counter", 0) + 1
        ctx.session.state["escalation_check_counter"] = loop_counter
        
        # Force escalation after 2 iterations to prevent token overflow
        if loop_counter >= 2:
            logging.warning(f"[{self.name}] Maximum iterations reached. Forcing escalation.")
            should_escalate = True
        
        if should_escalate:
            logging.info(f"[{self.name}] Escalating (grade: {grade_found}, iteration: {loop_counter})")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(f"[{self.name}] Continuing loop (grade: {grade_found}, iteration: {loop_counter})")
            yield Event(author=self.name)

# Optional: Add state cleanup agent to run at the end
class StateCleanupAgent(BaseAgent):
    """Cleans up intermediate state data to prevent token accumulation."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Clean up intermediate state data to free tokens."""
        try:
            # Remove large intermediate data but keep final outputs
            keys_to_clean = [
                "research_plan",
                "compact_research_data", 
                "gap_fill_research",
                "structured_research_data"
            ]
            
            cleaned_count = 0
            for key in keys_to_clean:
                if key in ctx.session.state:
                    ctx.session.state.pop(key, None)
                    cleaned_count += 1
            
            logging.info(f"[{self.name}] Cleaned {cleaned_count} intermediate state keys")
            
            # Keep only essential final outputs
            essential_keys = {"organizational_intelligence_agent", "org_html", "sources"}
            current_keys = set(ctx.session.state.keys())
            for key in current_keys - essential_keys:
                if key not in ["url_to_short_id", "escalation_check_counter"]:
                    ctx.session.state.pop(key, None)
            
            yield Event(author=self.name)
            
        except Exception as e:
            logging.error(f"[{self.name}] Error during cleanup: {e}")
            yield Event(author=self.name)

# Add cleanup to the pipeline
organizational_research_pipeline= SequentialAgent(
    name="organizational_research_pipeline_optimized",
    description="Token-optimized organizational intelligence pipeline with cleanup.",
    sub_agents=[
        organizational_section_planner,
        organizational_researcher,
        LoopAgent(
            name="quality_assurance_loop",
            max_iterations=2,
            sub_agents=[
                organizational_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_organizational_search,
            ],
        ),
        organizational_report_composer,
        org_html_composer
    ],
)

# --- MAIN ORGANIZATIONAL INTELLIGENCE AGENT ---
# organizational_intelligence_agent = LlmAgent(
#     name="organizational_intelligence_agent",
#     model=config.worker_model,
#     description="Advanced organizational intelligence system creating comprehensive reports for strategic sales intelligence.",
#     instruction=f"""
#     You are an advanced Organizational Intelligence System specializing in comprehensive company research and professional report generation for strategic sales and business development.

#     **CORE MISSION:**
#     Transform any organizational research request into a systematic intelligence gathering operation that produces a professional, citation-rich report.

#     **OPERATIONAL PROTOCOL:**
    
#     **Step 1: REQUEST ANALYSIS**
#     - Parse user request to identify target organization(s)
#     - Determine research scope and specific intelligence requirements
#     - Assess any special focus areas or constraints

#     **Step 2: STRATEGIC PLANNING**
#     - **MANDATORY:** Use `organizational_plan_generator` to create comprehensive research strategy
#     - Never attempt direct research without a systematic plan
#     - Ensure plan covers all critical business intelligence areas

#     **Step 3: RESEARCH EXECUTION**
#     - Delegate complete research execution to `organizational_research_pipeline`
#     - Monitor for quality assurance loop execution
#     - Ensure comprehensive data collection across all research phases

#     **RESEARCH INTELLIGENCE FOCUS:**
    
#     *Strategic Sales Intelligence:*
#     - Decision-maker identification and contact mapping
#     - Buying signal detection and opportunity timing
#     - Budget capacity and financial health assessment
#     - Competitive positioning and differentiation analysis

#     *Comprehensive Business Intelligence:*
#     - Corporate structure and leadership analysis
#     - Financial performance and market position
#     - Technology infrastructure and innovation focus
#     - Risk assessment and due diligence factors

#     *Market & Competitive Analysis:*
#     - Industry positioning and market share data
#     - Competitive landscape and threat assessment
#     - Strategic partnerships and alliance networks
#     - Recent developments and future strategic direction

#     **OUTPUT SPECIFICATIONS:**
#     - Professional report with Wikipedia-style citations
#     - Comprehensive coverage of all business intelligence areas
#     - Sales-ready insights and strategic recommendations
#     - Risk assessment and opportunity analysis
#     - Executive summary with key strategic insights

#     **QUALITY STANDARDS:**
#     - Multi-source verification for critical facts
#     - Recent information prioritized (12-18 months)
#     - Both positive and negative aspects included
#     - Professional presentation with proper citations
#     - Actionable intelligence for sales strategy

#     Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}

#     **REMEMBER:** Always begin with strategic planning, then execute through the comprehensive research pipeline.
#     """,
#     sub_agents=[organizational_research_pipeline],
#     tools=[AgentTool(organizational_plan_generator)],
#     output_key="organizational_intelligence_system",
# )