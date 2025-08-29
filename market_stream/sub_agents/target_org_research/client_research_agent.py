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

# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Model representing a specific search query for organizational research."""
    search_query: str = Field(
        description="A highly specific and targeted query for organizational web search, focusing on company information, social media, and public perception."
    )
    research_phase: str = Field(
        description="The research phase this query belongs to: 'foundation', 'market_intelligence', 'deep_dive', or 'risk_assessment'"
    )

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

# --- Enhanced Callbacks ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources with enhanced metadata."""
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
            title = chunk.web.title if chunk.web.title != chunk.web.domain else chunk.web.domain
            domain = chunk.web.domain or "unknown"
            
            # Handle cases where title might be None
            if not title or title == domain:
                title = domain or "Unknown Source"
            
            if url not in url_to_short_id:
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
            chunks_info[idx] = url_to_short_id[url]
        
        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        short_id = chunks_info[chunk_idx]
                        confidence = confidence_scores[i] if i < len(confidence_scores) else 0.5
                        text_segment = support.segment.text if support.segment else ""
                        sources[short_id]["supported_claims"].append({
                            "text_segment": text_segment,
                            "confidence": confidence,
                        })
    
    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources

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

def html_report_generator_callback(callback_context: CallbackContext) -> genai_types.Content:
    """Generates a polished HTML report with Wikipedia-style citations and professional styling."""
    report_content = callback_context.state.get("organizational_intelligence_report", "")
    sources = callback_context.state.get("sources", {})
    
    # Create citation mapping
    short_id_to_index = {}
    for idx, short_id in enumerate(sorted(sources.keys()), start=1):
        short_id_to_index[short_id] = idx

    # Replace citation tags with numbered superscript links
    def citation_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if short_id not in short_id_to_index:
            logging.warning(f"Invalid citation: {match.group(0)}")
            return ""
        index = short_id_to_index[short_id]
        return f'<sup><a href="#ref{index}" class="citation-link">[{index}]</a></sup>'

    # Process citations
    processed_content = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/?>',
        citation_replacer,
        report_content
    )
    
    # Generate references section
    references_html = "\n<h2 id='references'>References</h2>\n<ol class='references-list'>\n"
    for short_id, idx in sorted(short_id_to_index.items(), key=lambda x: x[1]):
        source = sources[short_id]
        source_type = source.get('source_type', 'Web')
        access_date = source.get('access_date', datetime.datetime.now().strftime("%Y-%m-%d"))
        
        references_html += (
            f'<li id="ref{idx}">'
            f'<a href="{source["url"]}" target="_blank" rel="noopener">{source["title"]}</a>. '
            f'<em>{source["domain"]}</em>. '
            f'<span class="source-type">[{source_type}]</span> '
            f'<span class="access-date">Retrieved {access_date}</span>'
            f'</li>\n'
        )
    references_html += "</ol>\n"
    
    # Create complete HTML document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Organizational Intelligence Report</title>
    <style>
        /* Professional Report Styling */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .report-header {{
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }}
        
        .report-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            opacity: 0.3;
        }}
        
        .report-header h1 {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }}
        
        .report-subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }}
        
        .report-meta {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 1px solid #e9ecef;
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .report-content {{
            padding: 40px;
        }}
        
        .table-of-contents {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 40px;
            border-left: 4px solid #667eea;
        }}
        
        .table-of-contents h2 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .toc-list {{
            list-style: none;
        }}
        
        .toc-list li {{
            margin: 8px 0;
        }}
        
        .toc-list a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }}
        
        .toc-list a:hover {{
            color: #764ba2;
            text-decoration: underline;
        }}
        
        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        h1 {{
            font-size: 2.2em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        h2 {{
            font-size: 1.8em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 8px;
        }}
        
        h3 {{
            font-size: 1.4em;
            color: #495057;
        }}
        
        h4 {{
            font-size: 1.2em;
            color: #6c757d;
        }}
        
        p {{
            margin-bottom: 16px;
            text-align: justify;
        }}
        
        .executive-summary {{
            border-radius: 8px;
            padding: 25px;
            margin: 20px 0;
            border-left: 4px solid #ff6b6b;
        }}
        
        .section-highlight {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .financial-metrics {{
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .risk-warning {{
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .key-insights {{
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        ul, ol {{
            margin: 16px 0 16px 30px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        .citation-link {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }}
        
        .citation-link:hover {{
            color: #764ba2;
        }}
        
        .references-list {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        
        .references-list li {{
            margin: 15px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .references-list a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }}
        
        .references-list a:hover {{
            text-decoration: underline;
        }}
        
        .source-type {{
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            margin: 0 5px;
        }}
        
        .access-date {{
            color: #6c757d;
            font-style: italic;
            font-size: 0.85em;
        }}
        
        .data-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .data-card {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .data-card h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #2c3e50;
            margin: 5px 0;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .report-header {{
                padding: 20px;
            }}
            
            .report-header h1 {{
                font-size: 1.8em;
            }}
            
            .report-content {{
                padding: 20px;
            }}
            
            .data-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Smooth scrolling */
        html {{
            scroll-behavior: smooth;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                background: white;
                font-size: 12pt;
            }}
            
            .report-container {{
                box-shadow: none;
                border-radius: 0;
            }}
            
            .report-header {{
                background: #667eea !important;
                color: white !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        {processed_content}
        {references_html}
    </div>
</body>
</html>"""
    
    callback_context.state["organizational_intelligence_agent"] = html_content
    return genai_types.Content(parts=[genai_types.Part(text=html_content)])

# --- Enhanced Agent Definitions ---
organizational_plan_generator = LlmAgent(
    model=config.search_model,
    name="organizational_plan_generator",
    description="Generates comprehensive sales intelligence research plans with exact name matching and detailed search strategies for targeted marketing campaigns.",
    instruction=f"""
    You are an expert sales intelligence strategist specializing in comprehensive target organization research for precision marketing and sales campaigns.

    **MISSION:** Create a systematic research plan to investigate target organizations, focusing on actionable sales intelligence with EXACT name matching to enable targeted product marketing.

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
    - Note any common name variations or legal entity names

    **RESEARCH METHODOLOGY - 5 PHASES:**

    **Phase 1: Organizational Foundation & Structure (20% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Official company website exploration (products/services, target markets, value proposition)
    - Organizational hierarchy and reporting structure analysis
    - Key departments and team structures relevant to our product offerings
    - Company size, employee count, geographic distribution, and office locations
    - Business model analysis and revenue streams

    **Phase 2: Decision-Maker Identification & Mapping (25% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - C-suite executive team identification and background analysis
    - Department heads and key managers across relevant functions
    - Technical decision-makers, IT leadership, and technology influencers
    - Procurement, purchasing, and vendor management teams
    - Reporting relationships and influence networks

    **Phase 3: Technology Landscape & Infrastructure (20% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Current technology stack, software solutions, and infrastructure
    - Recent technology investments, upgrades, and modernization initiatives
    - Technology roadmap and future investment priorities
    - Compatibility assessment with our product offerings
    - Integration requirements and technical constraints

    **Phase 4: Business Needs & Pain Points (20% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Current business challenges, operational pain points, and inefficiencies
    - Growth initiatives, expansion plans, and strategic projects
    - Competitive pressures and market challenges
    - Regulatory compliance requirements and industry-specific needs
    - Customer satisfaction issues and service delivery gaps

    **Phase 5: Sales Intelligence & Buying Signals (15% effort):**
    Generate [RESEARCH] tasks with EXACT name matching for:
    - Budget cycles, planning processes, and spending patterns
    - Recent vendor relationships, partnerships, and solution providers
    - Competitor presence, market share, and solution adoption
    - Buying signals, trigger events, and opportunity timing indicators
    - Procurement processes and decision-making criteria

    **EXACT SEARCH STRATEGY GUIDELINES:**
    - Always use the complete organization name in quotation marks
    - Create specific, targeted search queries with exact name matching
    - Focus on recent information (last 12-18 months) for accuracy
    - Include both technology and business-focused searches
    - Prioritize authoritative sources (official sites, LinkedIn, industry publications)
    - Balance executive leadership research with technical decision-maker identification

    **OUTPUT FORMAT:**
    Structure your plan with clear phase divisions, specific research objectives, and actionable search strategies that maintain exact name matching throughout.
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    output_key="research_plan",
    tools=[google_search],
)

from .org_report_template import TEMPLATE

organizational_section_planner = LlmAgent(
    model=config.worker_model,
    name="organizational_section_planner",
    description="Creates detailed HTML report structure following the standardized organizational intelligence format.",
    instruction=TEMPLATE,
    output_key="report_sections",
)

organizational_researcher = LlmAgent(
    model=config.search_model,
    name="organizational_researcher",
    description="Deep-dive organizational intelligence researcher with systematic approach to company analysis.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a senior business intelligence researcher specializing in comprehensive organizational analysis for strategic sales intelligence.

    **CORE RESEARCH METHODOLOGY:**

    **1. SYSTEMATIC SEARCH EXECUTION WITH EXACT NAME MATCHING:**
    Execute searches in logical sequence, building knowledge progressively. CRITICAL: Always use the complete, exact organization name in quotation marks for precision.

    *Foundation Searches (Use EXACT company name in quotes):*
    - "\"[EXACT Company Name]\" official website about company"
    - "\"[EXACT Company Name]\" business model revenue how they make money"
    - "\"[EXACT Company Name]\" leadership team executives CEO CFO"
    - "\"[EXACT Company Name]\" company size employees headcount"
    - "\"[EXACT Company Name]\" headquarters locations offices"

    *Financial Intelligence (Use EXACT company name in quotes):*
    - "\"[EXACT Company Name]\" revenue financial performance 2024"
    - "\"[EXACT Company Name]\" funding series investors venture capital"
    - "\"[EXACT Company Name]\" stock price market cap public private"
    - "\"[EXACT Company Name]\" SEC filings 10-K annual report"

    *Market & Competitive Analysis (Use EXACT company name in quotes):*
    - "\"[EXACT Company Name]\" competitors competitive landscape"
    - "\"[EXACT Company Name]\" market share industry position"
    - "\"[EXACT Company Name]\" news recent developments 2024"
    - "\"[EXACT Company Name]\" partnerships strategic alliances"

    *Risk & Opportunity Assessment (Use EXACT company name in quotes):*
    - "\"[EXACT Company Name]\" controversies legal issues regulatory problems"
    - "\"[EXACT Company Name]\" growth opportunities expansion plans"
    - "\"[EXACT Company Name]\" customer reviews testimonials case studies"
    - "\"[EXACT Company Name]\" technology stack digital transformation"

    **EXACT NAME MATCHING PROTOCOL:**
    - NEVER truncate or abbreviate the organization name
    - ALWAYS use the complete name exactly as provided by the user
    - Use quotation marks around the full company name for precise matching
    - If no results are found with the exact name, note this explicitly
    - Only use alternative search terms if the exact name yields no results
    - Distinguish the target organization from similarly named companies
    - Verify you're researching the correct entity by checking website domains and official presence

    *Risk & Opportunity Assessment:*
    - "[Company Name] controversies legal issues regulatory problems"
    - "[Company Name] growth opportunities expansion plans"
    - "[Company Name] customer reviews testimonials case studies"
    - "[Company Name] technology stack digital transformation"

    **2. SOURCE PRIORITIZATION HIERARCHY:**
    - **Tier 1 (Authoritative):** Company official website, SEC filings, major financial databases
    - **Tier 2 (Professional):** LinkedIn profiles, industry publications, major news outlets
    - **Tier 3 (Contextual):** Business databases (Crunchbase), analyst reports, trade publications
    - **Tier 4 (Supplementary):** Social media, forums, review sites

    **3. INFORMATION VERIFICATION STANDARDS:**
    - Cross-reference critical facts across minimum 2 sources
    - Note conflicting information and source reliability
    - Prioritize recent information (12-18 months) with historical context
    - Flag unverified claims clearly in findings

    **4. COMPREHENSIVE DATA COLLECTION:**
    
    *Company Fundamentals:*
    - Legal entity name, DBA names, subsidiaries
    - Industry classification (NAICS/SIC codes)
    - Business model and revenue streams
    - Geographic presence and market focus
    - Company age, founding story, evolution

    *Financial Intelligence:*
    - Revenue figures (annual/quarterly if available)
    - Funding history and investor details
    - Profitability indicators and growth trends
    - Market valuation and financial health metrics
    - Budget allocation patterns and spending priorities

    *Leadership Analysis:*
    - C-suite executive profiles and backgrounds
    - Board composition and key stakeholders
    - Recent leadership changes and implications
    - Decision-making authority and procurement processes
    - Employee count and organizational structure

    *Market Position:*
    - Competitive landscape and direct competitors
    - Market share and industry ranking
    - Unique value propositions and differentiators
    - Customer base characteristics and segments
    - Partnership ecosystem and strategic alliances

    *Strategic Intelligence:*
    - Recent developments and strategic initiatives
    - Technology investments and digital maturity
    - Growth indicators and expansion signals
    - Innovation focus and R&D priorities
    - Merger, acquisition, or partnership activity

    *Risk Assessment:*
    - Regulatory compliance and legal issues
    - Reputation risks and public perception
    - Financial stability and operational challenges
    - Competitive threats and market vulnerabilities
    - Customer satisfaction and retention indicators

    **5. RESEARCH QUALITY STANDARDS:**
    - **Objectivity:** Include both positive and negative findings
    - **Recency:** Emphasize developments from last 12-18 months
    - **Relevance:** Focus on sales and business relationship implications
    - **Depth:** Provide specific examples and concrete evidence
    - **Attribution:** Clearly indicate source reliability and dates

    **OUTPUT REQUIREMENTS:**
    Compile comprehensive findings addressing all research areas with:
    - Specific facts with source attribution
    - Direct quotes from official sources when relevant
    - Quantitative data and metrics where available
    - Recent developments with dates and context
    - Both opportunities and risks identified
    - Sales-relevant intelligence highlighted

    Your research will feed into a professional HTML report - ensure thoroughness and accuracy.
    """,
    tools=[google_search],
    output_key="organizational_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

# Enhanced evaluator with stricter standards
organizational_evaluator = LlmAgent(
    model=config.critic_model,
    name="organizational_evaluator",
    description="Rigorous evaluation specialist for organizational intelligence research completeness and quality.",
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
    description="Targeted gap-filling researcher executing precision searches to complete organizational intelligence.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a precision organizational researcher specializing in targeted intelligence gathering to address specific research gaps.

    **MISSION:** Execute focused follow-up research to elevate organizational intelligence to professional standards.

    **EXECUTION PROTOCOL:**

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
    *For Financial Information (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" 10-K SEC filing annual report"
    - "\"[EXACT Company Name]\" revenue earnings financial results 2024"
    - "\"[EXACT Company Name]\" Series A B C funding investors crunchbase"
    - "\"[EXACT Company Name]\" IPO public private market cap valuation"

    *For Leadership Intelligence (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" CEO name background LinkedIn profile"
    - "\"[EXACT Company Name]\" executive team leadership bios"
    - "\"[EXACT Company Name]\" board of directors advisors"
    - "\"[EXACT Company Name]\" org chart organizational structure"

    *For Competitive Analysis (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" vs competitors comparison analysis"
    - "\"[EXACT Company Name]\" market share industry leader"
    - "\"[EXACT Company Name]\" competitive advantages differentiation"
    - "\"[EXACT Company Name]\" industry report market research"

    *For Strategic Intelligence (ALWAYS use exact company name in quotes):*
    - "\"[EXACT Company Name]\" recent news acquisitions partnerships 2024"
    - "\"[EXACT Company Name]\" product launches new initiatives"
    - "\"[EXACT Company Name]\" expansion plans growth strategy"
    - "\"[EXACT Company Name]\" press releases corporate communications"

    **CRITICAL SEARCH PRECISION REQUIREMENTS:**
    - Replace [EXACT Company Name] with the complete organization name exactly as provided
    - Never abbreviate, truncate, or modify the organization name
    - Use quotation marks around the complete company name for every search
    - If searches with the exact name return limited results, document this rather than using partial names
    - Verify you're researching the correct organization by checking official domains and business registration

    **4. INFORMATION INTEGRATION:**
    - Seamlessly merge new findings with existing research
    - Resolve conflicts between sources with source hierarchy
    - Highlight newly discovered critical information
    - Maintain comprehensive coverage across all areas

    **5. QUALITY ENHANCEMENT:**
    - Verify key claims across multiple sources
    - Add specific metrics and quantitative data
    - Include recent developments with precise dates
    - Ensure sales-relevant insights are prominently featured

    **OUTPUT REQUIREMENTS:**
    Deliver enhanced organizational research findings that:
    - Address all gaps identified in the evaluation
    - Integrate seamlessly with previous research
    - Meet professional business intelligence standards
    - Provide actionable sales intelligence insights
    - Include proper source attribution for new information

    Focus on quality over quantity - each new piece of information should add significant value to the overall intelligence picture.
    """,
    tools=[google_search],
    output_key="organizational_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

organizational_report_composer = LlmAgent(
    model=config.critic_model,
    name="organizational_report_composer",
    include_contents="none",
    description="Expert business intelligence report writer creating comprehensive HTML organizational analysis reports.",
    instruction="""
    You are an expert business intelligence report writer specializing in comprehensive organizational analysis for strategic sales intelligence.

    **MISSION:** Transform research findings into a polished, professional HTML organizational intelligence report.

    ### INPUT DATA ANALYSIS
    **Research Data:** `{organizational_research_findings}`
    **Report Template:** `{report_sections}`
    **Citation Sources:** `{sources}`
    **Research Plan:** `{research_plan}`

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
    after_agent_callback=html_report_generator_callback,
)

# --- Enhanced Loop Control Agent ---
class EscalationChecker(BaseAgent):
    """Enhanced escalation checker with better evaluation detection and safety controls."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Improved escalation logic with multiple detection methods and safety controls."""
        
        evaluation_result = None
        
        try:
            # Method 1: Direct from session state
            evaluation_result = ctx.session.state.get("research_evaluation")
            if evaluation_result:
                logging.info(f"[{self.name}] Found evaluation in session state: {evaluation_result}")
            
            # Method 2: Search recent events for evaluation
            if not evaluation_result:
                for event in reversed(ctx.session.events[-10:]):  # Check last 10 events
                    if hasattr(event, 'author') and 'evaluator' in str(event.author).lower():
                        try:
                            # Try to parse evaluation from event content
                            content = str(event.content) if hasattr(event, 'content') else ""
                            if '"grade"' in content:
                                if '"pass"' in content.lower():
                                    evaluation_result = {"grade": "pass"}
                                    logging.info(f"[{self.name}] Found PASS in event content")
                                    break
                                elif '"fail"' in content.lower():
                                    evaluation_result = {"grade": "fail"}
                                    logging.info(f"[{self.name}] Found FAIL in event content")
                                    break
                        except Exception as e:
                            logging.warning(f"[{self.name}] Error parsing event: {e}")
                            continue
            
            # Method 3: Check all state values for evaluation objects
            if not evaluation_result:
                for key, value in ctx.session.state.items():
                    try:
                        if isinstance(value, dict) and "grade" in value:
                            evaluation_result = value
                            logging.info(f"[{self.name}] Found evaluation in state key '{key}'")
                            break
                        elif hasattr(value, 'grade'):
                            evaluation_result = {"grade": getattr(value, 'grade')}
                            logging.info(f"[{self.name}] Found evaluation object with grade attribute")
                            break
                    except Exception:
                        continue

        except Exception as e:
            logging.error(f"[{self.name}] Error during evaluation detection: {e}")

        # Determine escalation decision
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
            else:
                grade_text = str(evaluation_result).lower()
                if "pass" in grade_text:
                    grade_found = "pass"
                    should_escalate = True
                elif "fail" in grade_text:
                    grade_found = "fail"
                    should_escalate = False

        # Safety mechanism
        loop_counter = ctx.session.state.get("escalation_check_counter", 0) + 1
        ctx.session.state["escalation_check_counter"] = loop_counter
        
        # Log decision details
        logging.info(f"[{self.name}] Escalation Decision - Grade: {grade_found}, "
                    f"Should Escalate: {should_escalate}, Loop Counter: {loop_counter}")
        
        if should_escalate:
            logging.info(f"[{self.name}] Research quality APPROVED (grade: {grade_found}). Escalating to complete research.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        elif loop_counter >= 3:
            logging.warning(f"[{self.name}] Maximum iterations reached ({loop_counter}). "
                          f"Forcing escalation to prevent infinite loop.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(f"[{self.name}] Research needs improvement (grade: {grade_found}). "
                        f"Continuing loop iteration {loop_counter}.")
            yield Event(author=self.name)

# --- ENHANCED PIPELINE ASSEMBLY ---
organizational_research_pipeline = SequentialAgent(
    name="organizational_research_pipeline",
    description="Comprehensive organizational intelligence research pipeline with quality assurance loop and HTML report generation.",
    sub_agents=[
        organizational_section_planner,
        organizational_researcher,
        LoopAgent(
            name="quality_assurance_loop",
            max_iterations=3,  # Allow up to 3 iterations for quality improvement
            sub_agents=[
                organizational_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_organizational_search,
            ],
        ),
        organizational_report_composer,
    ],
)

# --- MAIN ORGANIZATIONAL INTELLIGENCE AGENT ---
organizational_intelligence_agent = LlmAgent(
    name="organizational_intelligence_agent",
    model=config.worker_model,
    description="Advanced organizational intelligence system creating comprehensive HTML reports for strategic sales intelligence.",
    instruction=f"""
    You are an advanced Organizational Intelligence System specializing in comprehensive company research and professional HTML report generation for strategic sales and business development.

    **CORE MISSION:**
    Transform any organizational research request into a systematic intelligence gathering operation that produces a professional, citation-rich HTML report.

    **OPERATIONAL PROTOCOL:**
    
    **Step 1: REQUEST ANALYSIS**
    - Parse user request to identify target organization(s)
    - Determine research scope and specific intelligence requirements
    - Assess any special focus areas or constraints

    **Step 2: STRATEGIC PLANNING**
    - **MANDATORY:** Use `organizational_plan_generator` to create comprehensive research strategy
    - Never attempt direct research without a systematic plan
    - Ensure plan covers all critical business intelligence areas

    **Step 3: RESEARCH EXECUTION**
    - Delegate complete research execution to `organizational_research_pipeline`
    - Monitor for quality assurance loop execution
    - Ensure comprehensive data collection across all research phases

    **RESEARCH INTELLIGENCE FOCUS:**
    
    *Strategic Sales Intelligence:*
    - Decision-maker identification and contact mapping
    - Buying signal detection and opportunity timing
    - Budget capacity and financial health assessment
    - Competitive positioning and differentiation analysis

    *Comprehensive Business Intelligence:*
    - Corporate structure and leadership analysis
    - Financial performance and market position
    - Technology infrastructure and innovation focus
    - Risk assessment and due diligence factors

    *Market & Competitive Analysis:*
    - Industry positioning and market share data
    - Competitive landscape and threat assessment
    - Strategic partnerships and alliance networks
    - Recent developments and future strategic direction

    **OUTPUT SPECIFICATIONS:**
    - Professional HTML report with Wikipedia-style citations
    - Comprehensive coverage of all business intelligence areas
    - Sales-ready insights and strategic recommendations
    - Risk assessment and opportunity analysis
    - Executive summary with key strategic insights

    **QUALITY STANDARDS:**
    - Multi-source verification for critical facts
    - Recent information prioritized (12-18 months)
    - Both positive and negative aspects included
    - Professional presentation with proper citations
    - Actionable intelligence for sales strategy

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}

    **REMEMBER:** Always begin with strategic planning, then execute through the comprehensive research pipeline.
    """,
    sub_agents=[organizational_research_pipeline],
    tools=[AgentTool(organizational_plan_generator)],
    output_key="organizational_intelligence_system",
)