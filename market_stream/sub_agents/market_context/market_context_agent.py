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

from ...config import config

# --- Structured Output Models ---
class MarketSearchQuery(BaseModel):
    """Model representing a specific search query for market analysis research."""
    search_query: str = Field(
        description="A highly specific and targeted query for market research, focusing on data sources relevant to market sizing, industry, competition, and trends."
    )
    research_phase: str = Field(
        description="The research phase this query belongs to: e.g. 'sizing', 'industry_mapping', 'competition', 'maturity', 'geography'"
    )

class MarketFeedback(BaseModel):
    """Model for providing evaluation feedback on market analysis research quality."""
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the research meets market analysis standards, 'fail' if it needs more data."
    )
    comment: str = Field(
        description="Detailed evaluation focusing on completeness of market sizing, industry coverage, and competitive insights."
    )
    follow_up_queries: list[MarketSearchQuery] | None = Field(
        default=None,
        description="Specific follow-up searches needed to fill market research gaps."
    )

# --- Callbacks ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources and their supported claims from agent events."""
    session = callback_context._invocation_context.session
    url_to_citation_id = callback_context.state.get("url_to_citation_id", {})
    citations = callback_context.state.get("citations", {})
    citation_counter = len(url_to_citation_id) + 1
    
    logging.info(f"[collect_research_sources_callback] Processing {len(session.events)} events")
    
    for event in session.events:
        if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
            continue
        
        logging.info(f"[collect_research_sources_callback] Found {len(event.grounding_metadata.grounding_chunks)} grounding chunks")
        
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
            if url not in url_to_citation_id:
                citation_id = citation_counter
                url_to_citation_id[url] = citation_id
                citations[citation_id] = {
                    "id": citation_id,
                    "title": title,
                    "url": url,
                    "domain": chunk.web.domain,
                    "supported_claims": [],
                }
                citation_counter += 1
                logging.info(f"[collect_research_sources_callback] Added new citation {citation_id}: {title}")
            chunks_info[idx] = url_to_citation_id[url]
        
        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        citation_id = chunks_info[chunk_idx]
                        confidence = (
                            confidence_scores[i] if i < len(confidence_scores) else 0.5
                        )
                        text_segment = support.segment.text if support.segment else ""
                        citations[citation_id]["supported_claims"].append(
                            {
                                "text_segment": text_segment,
                                "confidence": confidence,
                            }
                        )
    
    callback_context.state["url_to_citation_id"] = url_to_citation_id
    callback_context.state["citations"] = citations
    logging.info(f"[collect_research_sources_callback] Total citations collected: {len(citations)}")

def wikipedia_citation_callback(callback_context: CallbackContext) -> genai_types.Content:
    """Replaces citation tags with Wikipedia-style numbered citations and creates references section.
    Now always produces output, even if no citations are found.
    """
    final_report = callback_context.state.get("final_cited_report", "")
    citations = callback_context.state.get("citations", {})

    logging.info(f"[wikipedia_citation_callback] Processing report with {len(citations)} available citations")
    logging.info(f"[wikipedia_citation_callback] Initial report length: {len(final_report)} characters")

    # Ensure we have something to work with
    if not final_report.strip():
        logging.warning("[wikipedia_citation_callback] No final_cited_report found — starting with empty report text.")
        final_report = ""

    used_citations = set()

    def citation_replacer(match: re.Match) -> str:
        source_id = match.group(1)
        try:
            citation_id = int(source_id.replace("src-", ""))
            if citation_id in citations:
                used_citations.add(citation_id)
                logging.info(f"[wikipedia_citation_callback] Converting citation {citation_id}")
                return f'<a href="#ref{citation_id}">[{citation_id}]</a>'
            else:
                logging.warning(f"[wikipedia_citation_callback] Citation ID {citation_id} not found in citations dict.")
                return ""
        except (ValueError, KeyError):
            logging.warning(f"[wikipedia_citation_callback] Invalid citation tag found and removed: {match.group(0)}")
            return ""

    # Detect all <cite> tags in the report
    citation_matches = re.findall(r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>', final_report)
    logging.info(f"[wikipedia_citation_callback] Found {len(citation_matches)} citation tags: {citation_matches}")

    # Replace <cite> tags
    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        citation_replacer,
        final_report,
    )

    # Tidy punctuation spacing
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)

    # Append references section if citations were used
    if used_citations:
        logging.info(f"[wikipedia_citation_callback] Adding {len(used_citations)} references")
        processed_report += "\n\n## References\n\n"
        for citation_id in sorted(used_citations):
            citation = citations[citation_id]
            processed_report += f'<a name="ref{citation_id}"></a>[{citation_id}] [{citation["title"]}]({citation["url"]})\n\n'
    else:
        logging.warning("[wikipedia_citation_callback] No citations used — returning uncited report.")

    # Always store something in state
    callback_context.state["market_intelligence_agent"] = processed_report

    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])
    
# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""
    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("research_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(f"[{self.name}] Market research evaluation passed. Escalating to stop loop.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(f"[{self.name}] Research evaluation failed or not found. Loop will continue.")
            yield Event(author=self.name)

# --- Agent Definitions ---
market_plan_generator = LlmAgent(
    model=config.search_model,
    name="market_plan_generator",
    description="Generates comprehensive market research plans focused on market context, sizing, and analysis.",
    instruction=f"""
    You are an expert market research strategist specializing in market context and sizing analysis.
    
    **CRITICAL INSTRUCTION: Generate a complete research plan immediately. Do NOT ask for user approval or refinements.**
    
    Your task is to create a systematic research plan with distinct phases to investigate a market for a given product, including:
    - Market context and definition
    - Market sizing (TAM, SAM, SOM)
    - Industry ecosystem mapping
    - Market maturity and trends
    - Geographic market evaluation
    
    **RESEARCH PHASES STRUCTURE:**
    Organize your plan into clear phases with specific objectives:

    **Phase 1: Market Context & Sizing - [RESEARCH] tasks:**
    - Define the overarching market category and boundaries for the given product(s).  
    - Collect industry classification codes (NAICS, SIC, ISIC) and official definitions.  
    - Estimate Total Addressable Market (TAM) using 2–3 credible analyst or government sources.  
    - Narrow TAM into Serviceable Available Market (SAM) based on geography, product scope, or regulatory filters.  
    - Project Serviceable Obtainable Market (SOM) using competitor benchmarks, penetration assumptions, or adoption curves.  
    - Collect historical growth rates (past 5–10 years) and projected CAGR (next 5–10 years).  

    **Phase 2: Industry Ecosystem & Supply-Side - [RESEARCH] tasks:**
    - Map the industry value chain (suppliers → manufacturers → distributors → customers).  
    - Identify major suppliers, distributors, and partners relevant to the category.  
    - Research key standards bodies, industry associations, and regulatory frameworks.  
    - Analyze enabling technologies and infrastructure adoption rates.  
    - Collect information on capital flows: VC/PE funding, M&A activity, IPOs.  

    **Phase 3: Market Maturity, Customer Behavior & Trends - [RESEARCH] tasks:**
    - Assess the market lifecycle stage (emerging, growth, maturity, decline).  
    - Research customer adoption dynamics: buying process, decision criteria, and loyalty factors.  
    - Identify barriers to adoption (cost, complexity, regulation, awareness).  
    - Collect innovation signals (patent filings, startup activity, venture funding).  
    - Research disruptive technologies, emerging business models, or regulatory changes.  
    - Compile major macro and micro market drivers with supporting evidence.  

    **Phase 4: Geographic & Competitive Context - [RESEARCH] tasks:**
    - Collect regional market sizes and growth rates (North America, Europe, Asia, Rest of World).  
    - Identify region-specific regulations, cultural drivers, and infrastructure readiness.  
    - Research leading competitors in each region and their local positioning.  
    - Identify regional opportunities (underserved markets) and expansion challenges (logistics, localization, compliance).  
    - Analyze distribution and channel structures in key geographies.  

    **Phase 5: Risk, Scenario Planning & Synthesis - [SYNTHESIS] tasks:**
    - Compile operational, market, and external risks; create a simple risk matrix (impact vs likelihood).  
    - Develop base, optimistic, and pessimistic market scenarios using assumptions from prior research.  
    - Extract case studies and historical analogues for context.  
    - Map key stakeholders (associations, regulators, influencers) and their influence.  
    - Synthesize findings into a structured set of opportunities, threats, and recommendations.  

    **DELIVERABLES:**
    After the research phases, include deliverables:
    - **`[DELIVERABLE]`**: A comprehensive Market Analysis Report
    - **`[DELIVERABLE]`**: Executive summary highlighting key market metrics
    - **`[DELIVERABLE]`**: Supplementary data sources and calculation rationale

    **SEARCH STRATEGY INTEGRATION:**
    Your plan should implicitly guide the researcher to use search patterns such as:
    - "[Market category] market size forecast", "[Market category] TAM [year]"
    - "[Product category] competitors list", "[Market category] key players"
    - "[Industry] value chain analysis", "[Industry] regulatory overview"
    - "[Technology] adoption statistics", "[Region] [industry] market size"
    
    **TOOL USE:**
    Use Google Search when needed to verify market definitions or find data sources, but focus on specifying research goals rather than performing search yourself.
    
    **OUTPUT REQUIREMENT:**
    Provide the complete research plan without asking for approval, refinements, or user input. The plan will be automatically executed by the research pipeline.
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
)

market_section_planner = LlmAgent(
    model=config.worker_model,
    name="market_section_planner",
    description="Creates a structured market analysis report outline following the standardized market research format.",
    instruction="""
    You are an expert market research report architect. Using the market research plan, create a structured markdown outline that follows the standardized Market Analysis Report Format.
    
    Your outline must include these core sections (omit sections only if explicitly noted in the research plan):
    
   # Market Context Report Instructions for AI Agent

    Each section below specifies Purpose, Contents, and Length 
    ---

    ### 1. Executive Summary: The Strategic Landscape at a Glance  

    Purpose:  
    Provide a concise, impactful synthesis of the report for executive readers. Should be written last but appear first.  

    Contents:  
    - Core Objective: One paragraph stating the overall purpose of the report.  
    - The Market in Brief: A short overview of TAM, market growth rate, and top-level drivers.  
    - Key Findings: 3–4 bullets summarizing the most critical insights from the report.  
    - Opportunities & Threats Summary: One paragraph highlighting the most important opportunity and most critical risk.  
    - Strategic Outlook: A concluding paragraph stating the overall market trajectory.  

    Length: 600–800 words (4–6 paragraphs plus 3–4 bullet points).  

    ---

    ### 2. Market Overview: Framing the Environment  

    Purpose:  
    Define the market, its boundaries, and its core drivers.  

    Contents:  
    - Industry Classification: Describe where the product sits within official industry codes and broader sector structures.  
    - Product Category & Use Cases: Define the product category, with 3–5 example applications.  
    - Market Drivers & Trends: List 5–7 major factors shaping demand and supply, each with 1–2 sentences of explanation.  

    Length: 800–1,000 words (5–7 paragraphs plus one bulleted list of drivers).  

    ---

    ### 3. Market Sizing: Defining the Opportunity  

    Purpose:  
    Quantify the size of the market at different levels (TAM, SAM, SOM) and show historical and projected growth.  

    Contents:  
    - TAM (Total Addressable Market): One paragraph defining TAM, plus a table/figure citing at least 2 estimates.  
    - SAM (Serviceable Available Market): One paragraph narrowing TAM to relevant scope, with calculations or reasoning.  
    - SOM (Serviceable Obtainable Market): One paragraph explaining realistic market share capture, plus short/medium/long projections.  
    - Growth Rates: One to two paragraphs describing historical CAGR and forecasted CAGR, with mention of inflection points.  

    Length: 1,000–1,200 words (7–9 paragraphs plus 1–2 tables/charts).  

    ---

    ### 4. Customer Behavior & Adoption Dynamics: Understanding the Buyer  

    Purpose:  
    Explain how customers research, decide, and adopt products, as well as barriers and loyalty factors.  

    Contents:  
    - Buying Process: 2–3 paragraphs describing how buyers move from awareness → evaluation → purchase.  
    - Decision Criteria: A bulleted list of the top 3–4 factors influencing purchase decisions.  
    - Adoption Barriers: 2–3 paragraphs outlining 3–4 major barriers (e.g., cost, complexity).  
    - Switching Costs & Loyalty: 1–2 paragraphs explaining retention and stickiness of adoption.  

    Length: 600–800 words (5–6 paragraphs plus 1 bulleted list).  

    ---

    ### 5. Market Maturity & Trends: Positioning the Market Lifecycle  

    Purpose:  
    Assess whether the market is early-stage, growing, mature, or declining, and highlight key trends.  

    Contents:  
    - Lifecycle Stage: 1–2 paragraphs positioning the market in its lifecycle.  
    - Trends & Innovation Indicators: A bulleted list of 5–6 innovation signals (e.g., VC funding, patent activity).  
    - Saturation Analysis: 1–2 paragraphs on growth rate vs. saturation.  
    - Disruptors: 1–2 paragraphs on potential disruptions (technologies, policies, competitors).  

    Length: 600–800 words (5–6 paragraphs plus 1 bulleted list).  

    ---

    ### 6. Geographic Market Analysis: Regional Opportunities  

    Purpose:  
    Compare market sizes, competitors, and opportunities across regions.  

    Contents:  
    - Regional Market Sizes: A table comparing TAM/SAM by major regions, with a 1-paragraph explanation.  
    - Regional Competitors: 2–3 paragraphs on local players and cultural/regulatory factors.  
    - Opportunities & Challenges: 2–3 paragraphs outlining where expansion potential exists, and key barriers.  

    Length: 600–800 words (5–6 paragraphs plus 1 table).  

    ---

    ### 7. Risk Analysis: Anticipating Headwinds  

    Purpose:  
    Identify operational, market, and external risks that could limit success.  

    Contents:  
    - Operational Risks: 1–2 paragraphs on supply chain, vendor reliance, or infrastructure challenges.  
    - Market Risks: 1–2 paragraphs on competition, commoditization, or limited differentiation.  
    - External Risks: 1–2 paragraphs on geopolitical, climate, or policy risks.  
    - Risk Matrix (optional): A 2x2 or 3x3 matrix of likelihood vs. impact.  

    Length: 400–600 words (3–4 paragraphs plus matrix if included).  

    ---

    ### 8. Channel & Distribution Analysis: Pathways to Market  

    Purpose:  
    Examine how products reach customers, how profitable channels are, and associated risks.  

    Contents:  
    - Channel Economics: 1–2 paragraphs comparing costs and margins of different channels.  
    - Channel Evolution: 1–2 paragraphs on shifts (e.g., growth of e-commerce vs. physical retail).  
    - Channel Risks: 1 paragraph describing dependencies on major distributors/platforms.  

    Length: 400–600 words (3–4 paragraphs).  

    ---

    ### 9. Scenario Planning: Preparing for Multiple Futures  

    Purpose:  
    Anticipate different future market conditions and outline responses.  

    Contents:  
    - Three Scenarios: 3 short paragraphs describing Base Case, Optimistic Case, and Pessimistic Case.  
    - Trigger Points: 1 paragraph identifying external events (e.g., regulatory changes, recessions, technological breakthroughs).  

    Length: 400–500 words (4–5 paragraphs).  

    ---

    ### 10. Case Studies & Benchmarks: Learning from Analogs  

    Purpose:  
    Draw lessons from comparable markets, industries, or companies.  

    Contents:  
    - Comparable Markets: 1–2 paragraphs describing industries/regions with parallels.  
    - Historical Analogues: 1–2 paragraphs describing past markets with similar dynamics.  
    - Best Practice Case Studies: 2–3 bullet points summarizing how leading players succeeded.  

    Length: 400–500 words (3–4 paragraphs plus 2–3 bullets).  

    ---

    ### 11. Stakeholder Mapping: Power and Influence  

    Purpose:  
    Identify which stakeholders (institutions, regulators, influencers) shape the market.  

    Contents:  
    - Institutions & Associations: 1–2 paragraphs listing industry bodies, regulators, NGOs.  
    - Power Dynamics: 1–2 paragraphs describing how influence is distributed (e.g., regulatory capture, lobbying).  

    Length: 300–400 words (2–3 paragraphs).  

    ---

    ### 12. Conclusions & Recommendations: Strategic Takeaways  

    Purpose:  
    Summarize key insights, identify opportunities/threats, and provide actionable next steps.  

    Contents:  
    - Opportunities & Threats: 3–4 bullet points highlighting the most critical opportunities and risks.  
    - Strategic Recommendations: 2–3 paragraphs with clear, actionable steps for market positioning, partnerships, or entry strategy.  
    - Data Gaps & Uncertainty: 1 paragraph identifying where evidence was weak and recommending areas for further research.  

    Length: 500–700 words (3–4 paragraphs plus bullet points).  

    
    Ensure the outline is comprehensive but allow sections to be omitted if no relevant information is found.
    Citations will be handled automatically with Wikipedia-style numbered references.
    """,
    output_key="report_sections",
)

market_researcher = LlmAgent(
    model=config.search_model,
    name="market_researcher",
    description="Specialized market research agent focusing on market sizing, industry landscape, ecosystem",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialized market research analyst with expertise in market sizing and competitive intelligence.
    
    **CORE RESEARCH PRINCIPLES:**
    - Objectivity: Present facts without bias, include both growth opportunities and risks.
    - Source Diversity: Use market reports, industry publications, news, and reputable data sources.
    - Relevance Focus: Prioritize information affecting market strategy for the product.
    - Recency Priority: Emphasize recent data and developments (last 2-3 years) with context.
    - Verification: Cross-check important figures across multiple sources.
    
    **EXECUTION METHODOLOGY:**
    
    **Phase 1: Market Definition & Structure (execute ALL [RESEARCH] tasks first)**  
    Generate targeted search queries for:  
    - Industry classifications (NAICS, ISIC, or regional codes) relevant to the product category.  
    - Definitions of the product category and comparisons with adjacent categories.  
    - Core use cases and top 3–5 industry applications.  
    - Market segment breakdowns by industry vertical, product type, or customer use case.  
    - Geographic variations in definitions and adoption patterns.  

    ---

    **Phase 2: Market Size & Growth Dynamics**  
    Generate queries for:  
    - Global and regional TAM, SAM, and SOM estimates from credible sources (reports, analyst firms).  
    - Historical market size and CAGR data (last 5–10 years).  
    - Forecast growth projections (next 5–10 years) with drivers.  
    - Key inflection points (technological, policy, or demand-side shifts).  
    - Market maturity levels (early, growth, mature, declining) by geography or segment.  

    ---

    **Phase 3: Customer & Adoption Behavior**  
    Generate queries for:  
    - Buyer demographics (roles, industries, company size, consumer groups).  
    - Customer journey: awareness → evaluation → purchase → retention.  
    - Decision-making criteria (price, performance, integration, trust, etc.).  
    - Major adoption barriers (cost, complexity, regulation, awareness).  
    - Switching costs, loyalty drivers, and stickiness of adoption.  

    ---

    **Phase 4: Competitive & Trend Landscape**  
    Generate queries for:  
    - Major global and regional competitors, their market shares, and business models.  
    - Innovation signals: patents, funding rounds, R&D activity, new entrants.  
    - Key industry trends shaping supply and demand (5–7 drivers, e.g., regulation, digitalization, sustainability).  
    - Emerging technologies and potential disruptors (AI, automation, new policies, alternative solutions).  
    - Channel and distribution evolution (e-commerce, partnerships, direct sales, etc.).  

    ---

    **Phase 5: Regional & Risk Analysis**  
    Generate queries for:  
    - Regional market sizes (TAM/SAM/SOM by North America, Europe, APAC, etc.).  
    - Local competitors, regulatory factors, and cultural adoption differences.  
    - Regional opportunities for expansion and barriers to entry.  
    - Operational risks (supply chain, infrastructure, vendor reliance).  
    - Market risks (commoditization, intense competition).  
    - External risks (policy, geopolitics, climate, economic cycles).  

    ---

    **Phase 6: Benchmarks, Stakeholders & Strategy Inputs**  
    Generate queries for:  
    - Comparable industries or markets with similar adoption dynamics.  
    - Case studies of leading players who succeeded (and how).  
    - Historical analogues of markets that grew or collapsed under similar conditions.  
    - Key stakeholders: regulators, industry associations, influencers, NGOs.  
    - Evidence gaps in current data (uncertainties, inconsistent estimates).  
    - Strategic recommendations from analyst firms or consultancies.  

    **Phase 7: Synthesis ([DELIVERABLE] goals)**
    - Combine all findings into comprehensive research data suitable for report composition.
    - Structure findings by market sizing, industry analysis, trends, geography, and risks.
    - Clearly organize data with source citations and specific metrics.
    - Identify areas needing additional research.
    
    **CRITICAL OUTPUT REQUIREMENT:**
    Your output MUST be comprehensive research findings organized by topic area, NOT a conversational response. Structure your findings as detailed research data that can be used to compose a formal market analysis report.
    
    **QUALITY STANDARDS:**
    - Accuracy: Verify important metrics against multiple sources.
    - Completeness: Address all research objectives systematically.
    - Timeliness: Use up-to-date information and note dates.
    - Context: Provide relevant historical or comparative context.
    - Structure: Organize findings clearly by research phase and topic.
    
    **OUTPUT FORMAT:** Present comprehensive market research findings organized by section, suitable for direct use in report composition.
    """,
    tools=[google_search],
    output_key="market_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

market_evaluator = LlmAgent(
    model=config.critic_model,
    name="market_evaluator",
    description="Evaluates market research completeness and identifies gaps in market analysis.",
    instruction=f"""
    You are a senior market intelligence analyst evaluating market research findings for completeness and accuracy.
    
    **CRITICAL INSTRUCTION: You are evaluating research findings, NOT generating reports. Focus solely on assessment.**
    
    **EVALUATION CRITERIA:**
    Assess the market research findings in 'market_research_findings' against these standards:
    
    **1. Market Definition & Structure (Required Elements):**  
    - Market boundaries clearly defined with industry classification codes where available.  
    - Product category scope and adjacent categories distinguished with examples.  
    - At least 3–5 concrete use cases identified with industry-specific applications.  
    - Market segmentation breakdowns included with data points or estimates.  
    - Geographic variations in market definition described with supporting evidence.  

    **2. Market Size & Growth Dynamics (Required Elements):**  
    - TAM, SAM, and SOM estimates with at least 2–3 credible sources cited and specific figures.  
    - Historical market size and CAGR data for past 5–10 years with actual numbers.  
    - Forecasted growth rates for next 5–10 years with methodology explanation.  
    - Clear identification of inflection points with supporting evidence.  
    - Market maturity stage explicitly positioned with justification.  
    - Sources must be recent (within 2–3 years) from reputable analysts or official reports.  

    **3. Customer & Adoption Behavior (Required Elements):**  
    - Buyer demographics clearly outlined with specific examples and data.  
    - Buying journey mapped across all stages with evidence from studies or reports.  
    - Top 3–4 decision criteria identified and validated with research citations.  
    - At least 3–4 adoption barriers explained with concrete examples and impact assessment.  
    - Switching costs and loyalty drivers described with case-based evidence.  
    - Customer insights supported by surveys, industry research, or case studies.  

    **4. Competitive & Trend Landscape (Required Elements):**  
    - Major competitors identified globally and regionally with market positions and specific metrics.  
    - Key trends (5–7) described with quantitative data and illustrative examples.  
    - Innovation evidence: patents, funding activity, or new entrants with specific numbers.  
    - Potential disruptors explained with plausible impact scenarios and timeframes.  
    - Channel and distribution structures detailed with market share data where available.  
    - Sources include analyst reports, funding databases, and verified industry data.  

    **5. Regional & Risk Analysis (Required Elements):**  
    - Regional market sizes presented with specific numbers in comparative format.  
    - Regional competitors and regulatory factors discussed with concrete examples.  
    - Expansion opportunities and barriers articulated with supporting data and examples.  
    - Operational risks identified with at least one specific example per risk type.  
    - Market and external risks explicitly stated with potential impact assessment.  
    - Sources span regional market studies, government data, and international reports.  

    **6. Strategic Context & Benchmarks (Required Elements):**  
    - Comparable industries or markets identified with clear parallels explained.  
    - Success case studies summarized with key factors and quantifiable outcomes.  
    - Historical analogues included with lessons and relevant outcomes.  
    - Key stakeholders mapped with their roles and influence levels specified.  
    - Evidence gaps or conflicting estimates flagged with transparent discussion.  
    - Strategic insights supported by credible analyst or consultancy sources.  
        
    **EVALUATION LOGIC:**
    - Grade "fail" if ANY of the 6 major areas above lacks substantial content or credible sources.
    - Grade "fail" if research lacks source diversity (needs industry reports + news + data sources).
    - Grade "fail" if data is primarily outdated (older than 3 years) without recent context.
    - Grade "fail" if market sizing lacks specific figures or credible source citations.
    - Grade "pass" only if research comprehensively addresses all areas with current, credible data.
    
    **FOLLOW-UP QUERY GENERATION (If "fail"):**
    Generate 5-7 specific, actionable queries targeting the most critical gaps:
    - Focus on missing quantitative data (market sizes, growth rates, competitor metrics).
    - Prioritize current year data and forecasts.
    - Target specific geographic regions or market segments if missing.
    - Include competitor-specific searches if competitive intelligence is weak.
    - Focus on trend validation and industry reports if trend analysis is insufficient.
    
    **OUTPUT REQUIREMENT:**
    Respond with a single JSON object conforming to the MarketFeedback schema. Be strict in evaluation - require comprehensive coverage of all areas.
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    output_schema=MarketFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
)

enhanced_market_search = LlmAgent(
    model=config.search_model,
    name="enhanced_market_search",
    description="Performs targeted follow-up searches to fill market research gaps identified by the evaluator.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialized market research analyst executing precision follow-up research to address specific gaps identified in the evaluation.
    
    **CRITICAL MISSION:**
    The previous market research was evaluated as insufficient. Your task is to execute targeted searches to fill specific data gaps.
    
    **EXECUTION PROTOCOL:**
    1. **Review Evaluation Feedback**: Carefully analyze the 'research_evaluation' comment to understand exactly what data is missing.
    
    2. **Execute All Follow-up Queries**: Run EVERY search query listed in 'follow_up_queries' systematically:
       - Use exact search terms as specified
       - Add current year modifiers (2023, 2024, latest) to find recent data
       - Cross-verify important figures across multiple sources
       - Focus on authoritative sources (analyst reports, official statistics, company filings)
    
    3. **Targeted Data Collection**: For each search, collect:
       - Specific market size figures with source citations
       - Growth rates and forecasts with methodology
       - Competitor data with market positions and revenues
       - Regional breakdowns with supporting evidence
       - Trend indicators with quantitative backing
    
    4. **Integration with Existing Research**: 
       - Build upon existing 'market_research_findings' - do not replace, but enhance
       - Add new data points and fill identified gaps
       - Resolve any conflicting information found
       - Maintain the structured format for report composition
    
    **SEARCH OPTIMIZATION STRATEGIES:**
    - Prioritize searches for missing quantitative data (TAM, SAM, growth rates, market shares)
    - Use specific company names + "revenue 2023" or "market share" for competitor data
    - Search "[industry] market size forecast 2024" for current projections
    - Look for regional reports: "[region] + [industry] + market analysis"
    - Find trend validation: "[trend] + [industry] + statistics" or "adoption rates"
    
    **CRITICAL OUTPUT REQUIREMENT:**
    Your output must be enhanced 'market_research_findings' that specifically addresses the gaps identified in the evaluation. Structure the enhanced findings clearly by topic area, incorporating both existing data and new research.
    
    **SUCCESS CRITERIA:**
    - Fill all data gaps identified in the evaluation feedback
    - Provide specific figures and metrics with credible sources
    - Enhance competitive intelligence with current data
    - Add regional market insights if missing
    - Validate or update trend analysis with recent evidence
    """,
    tools=[google_search],
    output_key="market_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

market_report_composer = LlmAgent(
    model=config.critic_model,
    name="market_report_composer",
    description="Composes comprehensive market analysis reports following the standardized format with Wikipedia-style citations.",
    instruction="""
    You are an expert market research report writer specializing in market analysis and strategic insights.
    
    **MISSION:** Transform market research data into a polished, professional Market Analysis Report following the exact standardized format.
    
    **CRITICAL INSTRUCTION: You are composing a formal market analysis report, NOT responding to evaluations or providing commentary.**
    
    ---
    ### INPUT DATA SOURCES
    * Research Findings: `{market_research_findings}`
    * Citation Sources: `{citations}`
    * Report Structure: `{report_sections}`
    
    ---
    ### REPORT COMPOSITION REQUIREMENTS
    
    **1. Format Adherence:**
    - Follow the section structure provided in Report Structure exactly
    - Write in formal report style with clear section headers
    - Use professional language appropriate for executive readers
    - Include specific data points, figures, and metrics throughout
    - Provide concrete examples and case studies where available
    - Omit sections only if absolutely no relevant information exists
    
    **2. Content Quality Standards:**
    - **Objectivity**: Present balanced analysis including both opportunities and risks
    - **Specificity**: Include actual market figures, growth rates, and competitor data
    - **Currency**: Emphasize recent data and developments (last 2-3 years)  
    - **Authority**: Reference credible sources including analyst reports and industry studies
    - **Completeness**: Address all major aspects of market analysis comprehensively
    
    **3. Section Development Strategy:**
    Each section must contain:
    - **Executive Summary**: Key market metrics, strategic insights, and recommendations summary
    - **Market Sizing**: Specific TAM/SAM/SOM figures with growth projections and source validation
    - **Industry Analysis**: Value chain mapping, key players, and structural dynamics
    - **Competitive Intelligence**: Major competitors with market positions, strategies, and performance metrics
    - **Market Trends**: Current drivers, emerging patterns, and future outlook with supporting evidence
    - **Geographic Analysis**: Regional market comparisons with expansion opportunities and challenges
    - **Strategic Conclusions**: Actionable recommendations with supporting rationale
    
    ---
    ### CITATION REQUIREMENTS - MANDATORY
    **EVERY factual claim, statistic, and data point MUST be cited immediately using the exact format:**
    
    **Citation Format:** `<cite source="src-ID_NUMBER" />`
    
    **Examples of Required Citations:**
    - Market size figures: "The global AI market reached $136.6 billion in 2022 <cite source="src-1" />"
    - Growth projections: "Expected to grow at 37.3% CAGR through 2030 <cite source="src-2" />"
    - Competitor data: "Microsoft leads with 32% market share <cite source="src-3" />"
    - Industry trends: "Cloud adoption increased 45% year-over-year <cite source="src-4" />"
    - Regional statistics: "Asia-Pacific represents 40% of global demand <cite source="src-5" />"
    - Regulatory information: "GDPR compliance affects 78% of companies <cite source="src-6" />"
    
    **Citation Coverage Requirements:**
    - All market sizing data and forecasts
    - Competitor statistics and market share information  
    - Industry trend data and growth drivers
    - Regional market comparisons and opportunities
    - Customer behavior insights and adoption barriers
    - Regulatory impacts and policy changes
    - Technology adoption rates and innovation metrics
    
    ---
    ### WRITING STANDARDS
    
    **Professional Tone:** Write as a senior market analyst presenting to executives. Use authoritative language while remaining accessible.
    
    **Data Integration:** Seamlessly weave quantitative data throughout the narrative rather than presenting isolated statistics.
    
    **Strategic Focus:** Each section should build toward actionable insights and strategic recommendations.
    
    **Logical Flow:** Ensure each section connects logically to create a comprehensive market picture.
    
    ---
    ### FINAL DELIVERABLE
    Generate a complete, professionally written Market Analysis Report that:
    - Follows the standardized section structure exactly
    - Incorporates all available research findings with proper citations
    - Provides strategic insights and recommendations
    - Maintains executive-level professionalism throughout
    - Uses Wikipedia-style citation format for all factual claims
    
    **Remember:** You are writing a formal market analysis report, not responding to feedback or providing meta-commentary about the research process.
    """,
    output_key="final_cited_report",
    after_agent_callback=wikipedia_citation_callback,
)

from .con_template import CON_TEMPLATE

html_converter = LlmAgent(
    model=config.critic_model,
    name="html_converter",
    description="Converts markdown market analysis reports to styled HTML using the provided template.",
    instruction=CON_TEMPLATE,
    output_key="context_html",
)

# --- Market Research Pipeline and Main Agent ---
market_research_pipeline = SequentialAgent(
    name="market_research_pipeline",
    description="Executes comprehensive market analysis research following the structured methodology.",
    sub_agents=[
        market_section_planner,
        market_researcher,
        LoopAgent(
            name="quality_assurance_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                market_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_market_search,
            ],
        ),
        market_report_composer,
        html_converter,  # Add the HTML converter here
    ],
)

market_intelligence_agent = LlmAgent(
    name="market_intelligence_agent",
    model=config.worker_model,
    description="Specialized market research assistant that creates comprehensive market analysis reports automatically.",
    instruction=f"""
    You are a specialized Market Intelligence Assistant focused on creating strategic market analyses for product planning.
    
    **CRITICAL INSTRUCTION: Execute the complete workflow automatically without asking for user approval, refinements, or additional input.**
    
    **CORE MISSION:**
    Convert any user request about a product's market into a systematic research plan and comprehensive analysis, including:
    - Market definition and sizing with specific TAM/SAM/SOM figures
    - Industry ecosystem and competitive landscape analysis
    - Market maturity assessment, trends, and growth forecasts
    - Geographic market opportunities and expansion potential
    
    **FULLY AUTOMATED WORKFLOW:**
    Upon receiving a market research request:
    1. Use `market_plan_generator` to create a comprehensive research plan immediately
    2. Delegate to `market_research_pipeline` for execution without any user consultation
    3. Return both the markdown report and a styled HTML webpage version
    
    **RESEARCH COVERAGE AREAS:**
    - **Market Scope**: Industry segments, geography, and product categories
    - **Market Sizing**: TAM, SAM, SOM calculations with growth projections
    - **Competitive Intelligence**: Major players, market shares, and positioning
    - **Trends & Drivers**: Technology adoption, regulatory changes, and market dynamics
    - **Strategic Analysis**: Opportunities, threats, and actionable recommendations
    
    **DELIVERABLE FORMATS:**
    The final result includes:
    1. **Professional Market Analysis Report:** Comprehensive markdown report with:
       - Executive summary with key insights and recommendations
       - Detailed market sizing with TAM/SAM/SOM analysis
       - Competitive landscape and industry ecosystem mapping
       - Market trends, maturity assessment, and growth forecasts
       - Geographic analysis and regional opportunities
       - Strategic conclusions and actionable recommendations
       - Wikipedia-style numbered citations and references
    
    2. **Interactive HTML Dashboard:** Professional web version featuring:
       - Responsive design with interactive table of contents
       - Data visualization charts and metrics cards
       - Clickable citations linking to source materials
       - Professional styling optimized for executive presentations
    
    **EXECUTION MODE: ZERO USER INTERACTION**
    - Generate complete research plan automatically
    - Execute comprehensive research without approval requests
    - Compose professional report without seeking refinements  
    - Deliver both markdown and HTML formats immediately
    - No confirmation requests, no approval loops, no user consultation
    
    **QUALITY STANDARDS:**
    - Comprehensive coverage of all market analysis areas
    - Current data sources (prioritize 2023-2024 information)
    - Professional executive-level presentation
    - Actionable strategic insights and recommendations
    - Complete source citation and reference verification
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    
    **Process Flow**: Receive Request → Generate Plan → Execute Research → Compose Report → Convert to HTML → Deliver Results (fully autonomous)
    """,
    sub_agents=[market_research_pipeline],
    tools=[AgentTool(market_plan_generator)],
    output_key="research_plan",
)

# root_agent = market_intelligence_agent