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

from .config import config

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

def citation_replacement_callback(callback_context: CallbackContext) -> genai_types.Content:
    """Replaces citation tags in a report with Markdown-formatted links."""
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            logging.warning(f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        display_text = source_info.get("title", source_info.get("domain", short_id))
        return f" [{display_text}]({source_info['url']})"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
    callback_context.state["final_report_with_citations"] = processed_report
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
    model=config.worker_model,
    name="market_plan_generator",
    description="Generates comprehensive market research plans focused on market context, sizing, and analysis.",
    instruction=f"""
    You are an expert market research strategist specializing in market context and sizing analysis.
    
    Your task is to create a systematic research plan with distinct phases to investigate a market for a given product, including:
    - Market context and definition
    - Market sizing (TAM, SAM, SOM)
    - Industry ecosystem mapping
    - Market maturity and trends
    - Geographic market evaluation
    
    **RESEARCH PHASES STRUCTURE:**
    Organize your plan into clear phases with specific objectives:

    **Phase 1: Market Context & Sizing (30% of effort) - [RESEARCH] tasks:**
    - Define the market category and scope for the product.
    - Estimate Total Addressable Market (TAM) using credible sources.
    - Calculate Serviceable Available Market (SAM) based on product scope.
    - Provide Serviceable Obtainable Market (SOM) projections and assumptions.
    - Outline market growth trends and historical context.

    **Phase 2: Industry & Ecosystem (20% of effort) - [RESEARCH] tasks:**
    - Map the industry value chain and ecosystem (suppliers, channels, customers).
    - Identify key industry players, standards, and associations.
    - Analyze technology adoption and regulatory environment in the industry.
    - Highlight any consolidation or M&A trends.

    **Phase 3: Market Maturity & Trends (25% of effort) - [RESEARCH] tasks:**
    - Determine market lifecycle stage (emerging, growth, mature, decline).
    - Analyze adoption curve and customer adoption rates.
    - Identify market drivers, restraints, and emerging technologies.
    - Project future market development and potential disruptors.

    **Phase 4: Geographic Analysis (25% of effort) - [RESEARCH] tasks:**
    - Evaluate market size and dynamics in key regions (e.g., North America, Europe, Asia).
    - Identify regional trends, regulatory factors, and customer differences.
    - Determine priority geographies and expansion opportunities.

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
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Generate a detailed market research plan with the structure above.
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
    
    # 1. Executive Summary
    - Market definition and scope
    - Key market size metrics (TAM, SAM, SOM) and growth rates
    - Major findings and insights summary
    
    # 2. Market Overview
    - Industry classification and segments
    - Product category definitions and use cases
    - Market drivers and trends
    
    # 3. Market Sizing
    - Total Addressable Market (TAM) estimate and sources
    - Serviceable Available Market (SAM) and methodology
    - Serviceable Obtainable Market (SOM) assumptions and projections
    - Historical and projected market growth rates
    
    # 4. Industry Ecosystem
    - Industry value chain and ecosystem map
    - Key suppliers, partners, and distribution channels
    - Regulatory environment and standards bodies
    - Technology adoption and infrastructure
    
    # 6. Market Maturity & Trends
    - Market lifecycle stage with evidence (e.g., adoption curves)
    - Key market trends and innovation indicators
    - Growth rate and saturation analysis
    - Potential disruptors and technology trends
    
    # 7. Geographic Market Analysis
    - Regional market sizes and key differences
    - Major regional competitors and local factors
    - Geographic opportunities and expansion challenges
    
    # 8. Conclusions & Recommendations
    - Major opportunities and threats for the product
    - Strategic recommendations and next steps
    - Data gaps and uncertainty considerations
    
    Ensure the outline is comprehensive but allow sections to be omitted if no information is found.
    Do not include a separate References section - citations will be inline.
    """,
    output_key="report_sections",
)

market_researcher = LlmAgent(
    model=config.worker_model,
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
    
    **Phase 1: Market Sizing (Execute ALL [RESEARCH] tasks first)**
    Generate targeted search queries for:
    - Market size (TAM) for the product category globally.
    - Serviceable market segments (SAM) relevant to product.
    - Growth rates and market forecasts.
    - Sources: market research reports, government data, industry analyses.
    
    **Phase 2: Industry & Ecosystem**
    Generate queries for:
    - Industry value chain analysis.
    - Key players in the ecosystem (suppliers, partners).
    - Regulatory and standards bodies information.
    - Market drivers and technological infrastructure.
    
    **Phase 3: Market Maturity & Trends**
    Generate queries for:
    - Industry growth rates and adoption statistics.
    - Market lifecycle stage analysis.
    - Innovation trends and disruptive technologies.
    - Investment and funding trends in the market.
    
    **Phase 4: Geographic Analysis**
    Generate queries for:
    - Market size by region (North America, Europe, Asia, etc.).
    - Regional market growth and forecasts.
    - Regional regulatory or cultural factors affecting adoption.
    
    **Phase 5: Synthesis ([DELIVERABLE] goals)**
    - Combine all findings into the structured report sections.
    - Highlight key insights, numbers, and trends for each section.
    - Clearly cite sources for all data and factual claims.
    - Note any data gaps or uncertainties.
    
    **QUALITY STANDARDS:**
    - Accuracy: Verify important metrics against multiple sources.
    - Clarity: Present data in clear bullet points.
    - Completeness: Address all research objectives.
    - Timeliness: Use up-to-date information and note dates.
    - Context: Provide relevant historical or comparative context.
    
    **OUTPUT:** The output should be the market research findings suitable for report composition.
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
    You are a senior market intelligence analyst evaluating a market research report for completeness and accuracy.
    
    **EVALUATION CRITERIA:**
    Assess the market research findings in 'market_research_findings' against these standards:
    
    **1. Market Sizing (30%):**
    - TAM, SAM, SOM estimates present and well-justified
    - Growth rates and forecasts clearly identified
    - Market segments defined and sized appropriately
    
    **2. Industry & Ecosystem (20%):**
    - Industry value chain and ecosystem structure described
    - Key players, suppliers, partners identified
    - Regulatory and technological context covered
    
    **4. Market Maturity & Trends (25%):**
    - Market lifecycle stage determined with evidence
    - Adoption and growth trends analyzed
    - Innovation and disruption factors considered
    
    **5. Geographic Analysis (25%):**
    - Regional market differences and key regions covered
    - Regional trends and regulatory factors
    - Geographic opportunities identified
    
    **CRITICAL EVALUATION RULES:**
    1. Grade "fail" if any core element above is missing or insufficient.
    2. Grade "fail" if research lacks source diversity (needs mix of industry reports, news, and data sources).
    3. Grade "fail" if data is outdated (older than 3 years) without recent context.
    4. Grade "pass" only if research provides a comprehensive, up-to-date market analysis.
    
    **FOLLOW-UP QUERY GENERATION:**
    If grading "fail", generate 5-7 specific follow-up queries targeting the most critical gaps:
    - Focus on missing TAM/SAM data, competitor information, or trend analysis.
    - Prioritize current year or forecasted market data.
    - Include queries to verify or update key metrics.
    
    Your response must be a single JSON object conforming to the MarketFeedback schema.
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    output_schema=MarketFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
)

enhanced_market_search = LlmAgent(
    model=config.worker_model,
    name="enhanced_market_search",
    description="Performs targeted follow-up searches to fill market research gaps identified by the evaluator.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialized market research analyst executing precision follow-up research to address specific gaps.
    
    **MISSION:**
    Your previous market research was graded as insufficient. Now:
    
    1. Review Evaluation Feedback: Analyze 'research_evaluation' to understand deficiencies in:
       - Market sizing and forecast data
       - Industry ecosystem details
       - Trend and maturity analysis
       - Geographic market insights
    
    2. Execute Targeted Searches: Run EVERY query in 'follow_up_queries' using:
       - Exact product category 
       - Recent year modifiers (e.g., 2023, 2024, latest)
       - Industry report sources and news outlets
       - Cross-verify information across multiple sites
    
    3. Integrate and Enhance: Combine new findings with existing 'market_research_findings' to produce:
       - More complete market sizing calculations
       - Additional competitor metrics and sources
       - Enhanced trend analysis with latest data
       - Regional market breakdowns
    
    **SEARCH OPTIMIZATION:**
    - Prioritize authoritative data sources (industry reports, official statistics).
    - Seek multiple validations of key figures.
    - Use specific queries for missing data (e.g., exact competitor + "revenue 2023").
    
    **OUTPUT:** Your output should be the updated 'market_research_findings' addressing all identified gaps.
    """,
    tools=[google_search],
    output_key="market_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

market_report_composer = LlmAgent(
    model=config.critic_model,
    name="market_report_composer",
    include_contents="none",
    description="Composes comprehensive market analysis reports following the standardized format with proper citations.",
    instruction="""
    You are an expert market research report writer specializing in market analysis and strategic insights.
    
    **MISSION:** Transform market research data into a polished, professional Market Analysis Report following the exact standardized format.
    
    ---
    ### INPUT DATA SOURCES
    * Research Plan: `{research_plan}`
    * Research Findings: `{market_research_findings}`
    * Citation Sources: `{sources}`
    * Report Structure: `{report_sections}`
    
    ---
    ### REPORT COMPOSITION STANDARDS
    
    **1. Format Adherence:**
    - Follow the section structure provided in Report Structure exactly.
    - Use clear section headers and bullet points for readability.
    - Include date ranges and data sources for all statistics.
    - Provide specific examples and figures where available.
    - Omit sections only if no relevant information is found.
    
    **2. Content Quality:**
    - Objectivity: Present both positive opportunities and potential risks.
    - Relevance: Focus on information relevant to the product's market and strategy.
    - Recency: Emphasize recent data and developments (last 2-3 years).
    - Source Diversity: Reflect information from industry reports, news, and data sources.
    - Accuracy: Verify and cite all factual claims and figures.
    
    **3. Strategic Insights:**
    Each section should conclude with actionable insights:
    - Executive Summary: Key market takeaways for stakeholders.
    - Market Sizing: Implications for market opportunity and business planning.
    - Industry Ecosystem: Strategic partnerships and dependencies.
    - Competitive Landscape: Strategic positioning against competitors.
    - Market Trends: Growth opportunities and potential threats.
    - Geographic Analysis: Focus regions and local strategies.
    - Conclusions: Recommendations for product market strategy.
    
    ---
    ### CITATION REQUIREMENTS
    **Citation Format:** Use ONLY `<cite source="src-ID_NUMBER" />` tags immediately after factual claims.
    - Cite all market size figures and forecasts.
    - Cite competitor statistics and market share data.
    - Cite industry trends, regulatory facts, and growth rates.
    - Do NOT include a separate References section; all citations must be inline.
    
    ---
    ### FINAL QUALITY CHECKS
    - Ensure the report follows the exact outline structure.
    - Verify all sections contain concrete data and sources.
    - Confirm balance between opportunities and challenges.
    - Ensure recent information is highlighted and cited.
    - Maintain professional, objective tone throughout.
    
    Generate a complete Market Analysis Report to inform strategic decision-making.
    """,
    output_key="final_cited_report",
    after_agent_callback=citation_replacement_callback,
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
    ],
)

market_intelligence_agent = LlmAgent(
    name="market_intelligence_agent",
    model=config.worker_model,
    description="Specialized market research assistant that creates comprehensive market analysis reports.",
    instruction=f"""
    You are a specialized Market Intelligence Assistant focused on creating strategic market analyses for product planning.
    
    **CORE MISSION:**
    Convert any user request about a product's market into a systematic research plan and analysis, including:
    - Market definition and sizing 
    - Industry ecosystem and technology context
    - Market maturity, trends, and forecasts
    - Geographic market opportunities
    
    **CRITICAL WORKFLOW RULE:**
    NEVER answer market questions directly. Your only first action is to use `market_plan_generator` to create a research plan.
    
    **Your Process:**
    1. **Plan Generation:** Use `market_plan_generator` to create a research plan covering all core objectives.
    2. **Plan Refinement:** Work with user to refine plan details as needed (industry, geography, focus areas).
    3. **Research Execution:** Once the plan is approved by the user, delegate to `market_research_pipeline` with the approved plan.
    
    **RESEARCH FOCUS AREAS:**
    - Market Scope: industry segments, geography
    - Market Sizing: TAM, SAM, SOM, growth
    - Trends: adoption, technology, regulation
    - Strategy: opportunities and threats
    
    **OUTPUT EXPECTATIONS:**
    The final result should be a detailed Market Analysis Report with inline citations and actionable insights.
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    
    Remember: Plan → Refine → Execute. Never perform research directly - delegate to the specialized pipeline agents.
    """,
    sub_agents=[market_research_pipeline],
    tools=[AgentTool(market_plan_generator)],
    output_key="research_plan",
)

root_agent = market_intelligence_agent
