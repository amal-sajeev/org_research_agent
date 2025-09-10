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
    You are an expert organizational intelligence strategist specializing in comprehensive company research for sales and business development.
    
    **MISSION:** Create a systematic research plan to investigate organizations, focusing on actionable business intelligence with EXACT name matching.

    **CRITICAL NAME MATCHING REQUIREMENTS:**
    - Always use the COMPLETE, EXACT organization name as provided by the user
    - Use quotation marks around the full company name in searches to ensure exact matching
    - Never truncate, abbreviate, or use partial company names
    - If the organization name contains multiple words, treat it as a single entity
    - Example: For "Global Knowledge Technologies" always search for "Global Knowledge Technologies", never "Global Knowledge"

    **INITIAL VERIFICATION STEP:**
    Before creating the research plan, perform a verification search to:
    - Confirm the exact organization exists with the provided name
    - Identify the correct company website and official presence
    - Distinguish from similarly named organizations
    - Note any common name variations or legal entity names (e.g., "Inc.", "LLC", "Ltd.")

    **RESEARCH METHODOLOGY - 4 PHASES:**

    **Phase 1: Foundation Research (35% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Official company website exploration (about, leadership, products/services)
    - LinkedIn company page and executive profiles analysis
    - Basic corporate structure and business model investigation
    - Industry classification and market segment identification
    - Company size, employee count, and geographic presence

    **Phase 2: Financial & Market Intelligence (25% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Revenue data, funding history, and financial performance
    - SEC filings, annual reports, and investor relations materials
    - Market share data and competitive positioning
    - Recent business news and media coverage analysis
    - Industry analyst reports and market research

    **Phase 3: Leadership & Strategic Intelligence (25% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Executive team backgrounds and career histories
    - Recent leadership changes and organizational restructuring
    - Strategic partnerships and business alliances
    - Technology investments and innovation initiatives
    - Customer testimonials and case studies

    **Phase 4: Risk & Opportunity Assessment (15% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Regulatory issues and legal challenges
    - Reputation risks and public perception analysis
    - Competitive threats and market vulnerabilities
    - Growth opportunities and expansion signals
    - Buying signals and decision-making indicators

    **EXACT SEARCH STRATEGY GUIDELINES:**
    - Always use the complete organization name in quotation marks
    - Create specific, targeted search queries with exact name matching
    - Focus on recent information (last 12-18 months)
    - Include both positive and negative information gathering
    - Prioritize authoritative sources (official sites, financial databases, major news outlets)
    - Balance breadth with depth of investigation
    - If no results found with exact name, note this explicitly rather than using partial matches

    **OUTPUT FORMAT:**
    Structure your plan with clear phase divisions, specific research objectives, and actionable search strategies that maintain exact name matching throughout.
    
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

        #Basic Organizational Information
        - **Company legal name and DBA names**
        - **Core mission statement**
        - **Current operational status** (Growth, Maturity, Transformation, Turnaround)
        - **Employee count**
        - **Core operational units**
        - **Geographic footprint**
        - **Industry sector**
        - **Founded date**
        - **Headquarters location**

        #1. Executive Summary Data Requirements

        ##Operational Effectiveness Assessment
        - **Process maturity indicators**
        - **Productivity metrics and benchmarks**
        - **Quality standards and certifications**
        - **Operational efficiency ratios**
        - **Input-to-output conversion metrics**
        - **Operational bottlenecks and constraints**

        ##Financial Vitality Assessment
        - **Revenue trends (3-5 year historical)**
        - **Profitability margins**
        - **Cash flow patterns**
        - **Debt-to-equity ratios**
        - **Investment capacity indicators**
        - **Business model sustainability metrics**
        - **Financial stability ratings**

        ##Strategic Coherence Assessment
        - **Organizational structure alignment**
        - **Strategy communication effectiveness**
        - **Goal alignment across departments**
        - **Cultural-strategic fit**
        - **Resource allocation to strategic priorities**
        - **Performance measurement systems**

        #2. Core Capabilities & Operational Model

        ##Value Creation Engine
        - **Primary products/services offered**
        - **Key value propositions**
        - **Customer value delivery methods**
        - **Revenue generation mechanisms**
        - **Key business processes**
        - **Value chain analysis**
        - **Service/product development capabilities**

        ##Core Competencies
        - **Unique capabilities vs. competitors**
        - **Proprietary technologies or methods**
        - **Intellectual property portfolio**
        - **Brand differentiation factors**
        - **Specialized knowledge areas**
        - **Competitive advantages sustainability**

        ##Operational Structure & Processes
        - **Organizational design type** (functional, matrix, product-based)
        - **Reporting structures and hierarchy**
        - **Decision-making frameworks**
        - **Key workflow processes**
        - **Cross-functional collaboration methods**
        - **Process automation levels**
        - **Operational bottlenecks**
        - **Structural advantages/disadvantages**

        #3. Financial Health & Resource Allocation

        ##Spending Patterns & ROI
        - **Capital expenditure allocation**
        - **Operational expenditure breakdown**
        - **R&D investment levels**
        - **Marketing/sales spend efficiency**
        - **Technology investment ROI**
        - **Human capital investment returns**
        - **Cost structure vs. industry norms**

        ##Resource Allocation Strategy
        - **Budget allocation to strategic priorities**
        - **Human resource distribution**
        - **Capital allocation decision-making**
        - **Resource reallocation capability**
        - **Investment in growth vs. maintenance**
        - **Resource efficiency metrics**

        ##Financial Metrics
        - **Revenue diversity and recurrence**
        - **Profit margin analysis**
        - **Cash flow and liquidity position**
        - **Operational efficiency ratios**
        - **Working capital management**
        - **Debt service capabilities**
        - **Investment grade ratings**

        #4. Human Capital & Leadership Analysis

        ##Workforce Composition
        - **Employee demographics and diversity**
        - **Skills distribution across organization**
        - **Experience levels and tenure**
        - **Critical roles and succession planning**
        - **Skills gaps identification**
        - **Talent acquisition effectiveness**
        - **Training and development programs**

        ##Culture & Engagement
        - **Employee satisfaction surveys**
        - **Engagement score trends**
        - **Retention and turnover rates**
        - **Cultural values and behaviors**
        - **Internal communication effectiveness**
        - **Recognition and reward systems**
        - **Work-life balance indicators**

        ##Leadership Effectiveness
        - **Leadership development programs**
        - **Strategic vision communication**
        - **Decision-making speed and quality**
        - **Leadership succession planning**
        - **Management span of control**
        - **Leadership diversity and inclusion**
        - **Change management capabilities**

        #5. Technology & Operational Infrastructure

        ##Technology Stack Maturity
        - **Core technology platforms**
        - **System integration capabilities**
        - **Digital transformation progress**
        - **Technology scalability assessment**
        - **Cybersecurity posture**
        - **Data management capabilities**
        - **Technical debt analysis**
        - **Technology vendor relationships**

        ##Operational Resilience
        - **Business continuity plans**
        - **Disaster recovery capabilities**
        - **Supply chain resilience**
        - **Quality management systems**
        - **Risk management frameworks**
        - **Capacity planning and management**
        - **Single points of failure identification**
        - **Performance monitoring systems**

        #6. Strategic Market Position

        ##Competitive Advantage
        - **Market positioning vs. competitors**
        - **Competitive differentiation factors**
        - **Barriers to entry in market**
        - **Switching costs for customers**
        - **Network effects and scale advantages**
        - **Cost advantage sources**
        - **Innovation capabilities**

        ##Brand & Reputation Equity
        - **Brand recognition metrics**
        - **Customer loyalty indicators**
        - **Net Promoter Score (NPS)**
        - **Market perception surveys**
        - **Social media sentiment**
        - **Industry awards and recognition**
        - **Stakeholder trust levels**
        - **Crisis management track record**

        #7. Cultural Assessment & Organizational Health

        ##Core Cultural Traits
        - **Stated values vs. observed behaviors**
        - **Decision-making culture**
        - **Risk tolerance levels**
        - **Innovation and creativity support**
        - **Collaboration vs. competition balance**
        - **Performance management culture**
        - **Diversity and inclusion practices**

        ##Adaptability & Learning
        - **Change management success rate**
        - **Learning and development investment**
        - **Innovation pipeline and processes**
        - **Failure tolerance and learning**
        - **Knowledge management systems**
        - **Continuous improvement practices**
        - **External partnership openness**

        #8. SWOT Analysis Components

        ##Strengths (Internal Positive)
        - **Unique competitive advantages**
        - **Strong financial performance**
        - **Excellent leadership team**
        - **Proprietary technology/IP**
        - **Strong brand reputation**
        - **Skilled workforce**
        - **Efficient operations**
        - **Strategic partnerships**

        ##Weaknesses (Internal Negative)
        - **Skills or capability gaps**
        - **Financial constraints**
        - **Operational inefficiencies**
        - **Technology limitations**
        - **Brand perception issues**
        - **Talent retention challenges**
        - **Geographic limitations**
        - **Regulatory compliance issues**

        ##Opportunities (External Positive)
        - **Market growth trends**
        - **Emerging technologies**
        - **Regulatory changes favoring business**
        - **New market segments**
        - **Partnership possibilities**
        - **Economic conditions**
        - **Consumer behavior shifts**
        - **Industry consolidation opportunities**

        ##Threats (External Negative)
        - **Competitive pressures**
        - **Economic downturns**
        - **Regulatory challenges**
        - **Technology disruption**
        - **Changing customer preferences**
        - **Supply chain vulnerabilities**
        - **Talent shortages**
        - **Cybersecurity risks**


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
    instruction=f"""
    You are a senior business intelligence quality assurance specialist with expertise in organizational research evaluation.

    **MISSION:** Evaluate research findings against professional intelligence standards for comprehensive company analysis.

    **EVALUATION FRAMEWORK - 100 POINT SCALE:**

    **1. Company Fundamentals (25 points):**
    - Company identification and basic information (5 pts)
    - Business model and revenue streams clarity (5 pts)
    - Industry classification and market focus (5 pts)
    - Geographic presence and company structure (5 pts)
    - Founding information and company evolution (5 pts)

    **2. Financial Intelligence (25 points):**
    - Revenue data and financial performance (8 pts)
    - Funding history and investor information (7 pts)
    - Market valuation and financial health (5 pts)
    - Growth trends and financial indicators (5 pts)

    **3. Leadership & Organizational Analysis (20 points):**
    - Executive team identification and backgrounds (8 pts)
    - Organizational structure and decision-makers (6 pts)
    - Recent leadership changes and implications (6 pts)

    **4. Market & Competitive Intelligence (15 points):**
    - Competitive landscape understanding (5 pts)
    - Market position and unique advantages (5 pts)
    - Recent strategic developments (5 pts)

    **5. Sales Intelligence Value (15 points):**
    - Buying signals and opportunity indicators (5 pts)
    - Decision-making process insights (5 pts)
    - Risk assessment and due diligence factors (5 pts)

    **GRADING STANDARDS:**
    - **PASS (75+ points):** Research meets professional intelligence standards
    - **FAIL (<75 points):** Significant gaps requiring additional investigation

    **CRITICAL SUCCESS FACTORS:**
    - Minimum 3 different source types represented
    - Recent information (within 12-18 months) included
    - Both positive and negative aspects covered
    - Specific facts and figures provided (not just generalizations)
    - Sales-relevant intelligence clearly identified

    **FOLLOW-UP QUERY GENERATION (if FAIL):**
    Generate EXACTLY 3 highly specific queries targeting the most critical gaps:
    - Focus on missing foundational information first
    - Target specific data points (financial, leadership, competitive)
    - Prioritize information with highest sales impact

    **OUTPUT FORMAT:**
    Provide detailed JSON response with:
    - Point-by-point evaluation against the 100-point framework
    - Specific examples of strengths and weaknesses
    - Clear rationale for pass/fail decision
    - Targeted follow-up queries if needed

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}

    **IMPORTANT:** Be thorough but fair. High-quality research should pass even if some niche areas are incomplete.
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

    **EXECUTION GUIDELINES**
    **1. GAP ANALYSIS:**
    - Review evaluation feedback in 'research_evaluation' for specific deficiencies
    - Identify the most critical missing information categories
    - Prioritize gaps with highest sales intelligence value

    **2. PRECISION SEARCH STRATEGY:**
    - Execute ALL queries provided in 'follow_up_queries' efficiently
    - Use advanced search techniques for deeper information discovery
    - Focus on authoritative and recent sources
    - Apply alternative search angles if initial queries yield limited results

    **3. SEARCH OPTIMIZATION TECHNIQUES WITH EXACT NAME MATCHING:**
      **3. SEARCH OPTIMIZATION TECHNIQUES WITH EXACT NAME MATCHING:**
    *EXAMPLES For Financial Information (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" 10-K SEC filing annual report"
    - "\"[EXACT Company Name]\" revenue earnings financial results 2024"

    *EXAMPLES For Leadership Intelligence (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" CEO name background LinkedIn profile"
    - "\"[EXACT Company Name]\" executive team leadership bios"
    - "\"[EXACT Company Name]\" board of directors advisors"

    *EXAMPLES For Competitive Analysis (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" vs competitors comparison analysis"
    - "\"[EXACT Company Name]\" market share industry leader"
    - "\"[EXACT Company Name]\" industry report market research"

    *For Strategic Intelligence (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" recent news acquisitions partnerships 2024"
    - "\"[EXACT Company Name]\" product launches new initiatives"
    - "\"[EXACT Company Name]\" press releases corporate communications"

    **CRITICAL SEARCH PRECISION REQUIREMENTS:**
    - Replace [EXACT Company Name] with the complete organization name exactly as provided
    - Never abbreviate, truncate, or modify the organization name
    - Use quotation marks around the complete company name for every search
    - If searches with the exact name return limited results, document this rather than using partial names
    - Verify you're researching the correct organization by checking of   icial domains and business registration

    **EXECUTION PROTOCOL:**
    1. Review evaluation feedback for specific missing information
    2. Execute ALL queries from 'follow_up_queries' efficiently
    3. Focus on filling identified gaps only
    4. Integrate findings with existing research data

    **OUTPUT:** Provide ONLY the new information found - do not repeat existing research.
    Use structured bullet points for efficiency. ENSURE the following rules:
    - Address all gaps identified in the evaluation
    - Provide actionable sales intelligence insights
    - Include proper source attribution for new information

    **TOKEN EFFICIENCY:**
    - Focus on facts and data points
    - Avoid lengthy descriptions except for details about people
    - Update existing research categories only
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

        ### REPORT COMPOSITION STANDARDS

        **1. CONTENT TRANSFORMATION:**
        Replace ALL placeholders in the HTML template with comprehensive, well-researched content:

        *Executive Summary Requirements:*
        - Company legal name, industry, founding date, headquarters
        - Key financial metrics (revenue, funding, employees, valuation)
        - Primary business model and market position
        - High-level sales opportunity assessment
        - 3-4 key strategic insights

        *Detailed Section Requirements:*
        - **Company Overview:** Business model, products/services, target markets, value proposition
        - **Financial Performance:** Revenue trends, funding history, financial health indicators
        - **Leadership Analysis:** Executive profiles, decision-makers, organizational structure
        - **Market Intelligence:** Competitive landscape, market position, industry dynamics
        - **Technology Profile:** Tech stack, innovation focus, digital maturity
        - **Strategic Developments:** Recent news, partnerships, initiatives, achievements
        - **Risk Assessment:** Business risks, reputation factors, regulatory concerns
        - **Sales Intelligence:** Buying signals, budget indicators, decision processes
        - **Recommendations:** Optimal approach, timing, stakeholder targeting

        **2. CITATION INTEGRATION:**
        **CRITICAL:** Use ONLY `<cite source="src-ID_NUMBER" />` format for citations
        - Cite ALL factual claims, financial data, and specific information
        - Place citations immediately after the relevant statement
        - Cite leadership information and organizational details
        - Cite financial metrics and market data
        - Cite recent developments and strategic information

        **3. CONTENT QUALITY STANDARDS:**

        *Objectivity & Balance:*
        - Present both positive and negative findings
        - Include competitive challenges alongside advantages
        - Note risks and opportunities equally
        - Provide evidence-based analysis without bias

        *Specificity & Detail:*
        - Include specific figures, dates, and metrics
        - Name key executives and their backgrounds
        - Detail recent developments with timeframes
        - Provide concrete examples and case studies

        *Sales Intelligence Focus:*
        - Highlight decision-maker identification
        - Emphasize buying signals and opportunity indicators
        - Include budget and financial capacity insights
        - Provide actionable approach recommendations

        *Professional Presentation:*
        - Use appropriate HTML styling classes from the template
        - Structure information with clear headings and subheadings
        - Employ data cards for metrics and key figures
        - Use highlight boxes for critical insights

        **4. SPECIALIZED SECTION GUIDANCE:**

        *Financial Performance Section:*
        - Populate data cards with specific metrics
        - Include revenue figures, funding rounds, valuation data
        - Show growth trends and financial stability indicators
        - Use .financial-metrics class for highlighting

        *Risk Assessment Section:*
        - Use .risk-warning class for serious concerns
        - Balance risks with mitigation factors
        - Include regulatory, market, and operational risks
        - Provide context for risk evaluation

        *Sales Intelligence Section:*
        - Use .key-insights class for critical sales information
        - Detail buying signals and opportunity timing
        - Include decision-maker mapping and influence analysis
        - Provide budget and procurement insights

        **5. FINAL QUALITY REQUIREMENTS:**
        - NO placeholder text should remain in final output
        - ALL sections must be populated with relevant content
        - Citations must be properly formatted and comprehensive
        - Report must be professionally structured and complete
        - Content must be actionable for sales strategy development

        **IMPORTANT:** Your output will be processed by the HTML callback to generate the final styled report. Ensure all content is complete and properly cited.

        Generate a comprehensive organizational intelligence report that enables informed strategic sales decision-making.
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