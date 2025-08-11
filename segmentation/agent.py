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
class SegmentationSearchQuery(BaseModel):
    """Model representing a specific search query for segmentation research."""
    search_query: str = Field(
        description="A specific and targeted query for segmentation research, focusing on data sources relevant to market and customer segments."
    )
    research_phase: str = Field(
        description="The research phase this query belongs to: e.g. 'segment_discovery', 'customer_analysis', 'attractiveness', 'fit'"
    )

class SegmentationFeedback(BaseModel):
    """Model for providing evaluation feedback on segmentation research quality."""
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the segmentation research is comprehensive, 'fail' if it needs more detail."
    )
    comment: str = Field(
        description="Detailed evaluation focusing on completeness of segment discovery, customer segment profiles, and prioritization insights."
    )
    follow_up_queries: list[SegmentationSearchQuery] | None = Field(
        default=None,
        description="Specific follow-up searches needed to fill segmentation research gaps."
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
            logging.info(f"[{self.name}] Segmentation research evaluation passed. Escalating to stop loop.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(f"[{self.name}] Research evaluation failed or not found. Loop will continue.")
            yield Event(author=self.name)

# --- Agent Definitions ---
segmentation_plan_generator = LlmAgent(
    model=config.worker_model,
    name="segmentation_plan_generator",
    description="Generates comprehensive segmentation research plans focusing on market segments and customer segments.",
    instruction=f"""
    You are an expert market and customer segmentation strategist specializing in defining target market segments and customer groups for product targeting strategy.

    Your task is to create a systematic research plan with distinct phases to investigate segmentation for a given product, including:
    - Market Segment Discovery and Industry Mapping
    - Customer Segment Identification and Profiling
    - Segment Evaluation (attractiveness, competitive context)
    - Segmentation Framework and Targeting Recommendations

    **RESEARCH PHASES STRUCTURE:**
    Organize your plan into clear phases with specific objectives:

    **Phase 1: Market Segment Discovery (30% of effort) - [RESEARCH] tasks:**
    - Identify all potential market segments and sub-segments relevant to the product.
    - Map industry categories and use cases by segment.
    - Create a comprehensive inventory of addressable market segments.
    - Analyze segment characteristics and geographic variations.

    **Phase 2: Customer Segment Identification (20% of effort) - [RESEARCH] tasks:**
    - Identify distinct customer groups within each market segment (demographics, firmographics).
    - Characterize customer behaviors, needs, and decision criteria per segment.
    - Determine roles and influence patterns (decision makers vs end users).
    - Document customer journey stages relevant to the product.

    **Phase 3: Segment Attractiveness & Evaluation (25% of effort) - [RESEARCH] tasks:**
    - Assess segment size, growth trajectory, and revenue potential.
    - Evaluate competitive intensity and market share by segment.
    - Analyze accessibility and barriers for each segment.
    - Conduct entry barrier and channel analysis per segment.

    **Phase 4: Prioritization & Targeting Strategy (25% of effort) - [RESEARCH] tasks:**
    - Develop segment evaluation criteria (size, fit, competition).
    - Score and rank segments by attractiveness and strategic fit.
    - Align segments with product capabilities and company goals.
    - Provide targeting recommendations for top segments.

    **DELIVERABLES:**
    After the research phases, include deliverables:
    - **`[DELIVERABLE]`**: A Market Segment Inventory listing all viable segments with key attributes.
    - **`[DELIVERABLE]`**: A Customer Segment Matrix detailing customer profiles per segment.
    - **`[DELIVERABLE]`**: Segment Prioritization Rankings and Framework for top 3-5 segments.
    - **`[DELIVERABLE]`**: Targeting Recommendations for priority segments.

    **SEARCH STRATEGY INTEGRATION:**
    Your plan should implicitly guide the researcher to use search patterns such as:
    - "target market segments for [product category]"
    - "[product type] customer segmentation demographics"
    - "[industry] customer needs analysis"
    - "[product] use cases by industry"
    - "buying process [customer segment] [product]"

    **TOOL USE:**
    Use Google Search when needed to verify segment definitions or find relevant data, but focus on specifying research goals rather than performing search yourself.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Generate a detailed segmentation research plan with the structure above.
    """,
    tools=[google_search],
)

segmentation_section_planner = LlmAgent(
    model=config.worker_model,
    name="segmentation_section_planner",
    description="Creates a structured segmentation analysis report outline following a standardized format.",
    instruction="""
    You are an expert market segmentation report architect. Using the segmentation research plan, create a structured markdown outline that follows the standardized Segmentation Analysis Report format.

    Your outline must include these core sections (omit sections only if explicitly noted in the research plan):

    # 1. Executive Summary
    - Overview of scope and goals
    - Key target segments identified
    - Summary of major findings (e.g., top segments, growth opportunities)
    - Strategic recommendations in brief

    # 2. Market Segment Overview
    - List of identified market segments and categories
    - Definitions and characteristics of each segment
    - Industry context for each segment

    # 3. Customer Segment Profiles
    - Demographic and firmographic profiles by segment
    - Behavioral and needs-based patterns for each customer group
    - Key decision makers and influencers in each segment

    # 4. Segment Attractiveness & Competition
    - Segment size and growth rate estimates
    - Competitive intensity and major competitors per segment
    - Accessibility, channels, and barriers analysis for segments

    # 5. Segment Prioritization & Strategic Fit
    - Criteria for evaluation (size, fit, competition, strategic alignment)
    - Ranking of top segments with rationale
    - Alignment of product features to segment needs

    # 6. Segmentation Framework & Recommendations
    - Clear definition and boundaries for each priority segment
    - Targeting recommendations and next steps per segment
    - Opportunities and potential risks for each segment

    # 7. Conclusions
    - Recap of strategic insights
    - Final recommendations for market targeting strategy

    Ensure the outline is comprehensive but allow sections to be omitted if no relevant information is found. Do not include a separate References section - citations will be inline.
    """,
    output_key="report_sections",
)

segmentation_researcher = LlmAgent(
    model=config.worker_model,
    name="segmentation_researcher",
    description="Specialized segmentation research agent focusing on market segments, customer segmentation, and strategic analysis.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialized market and customer segmentation analyst.

    **CORE RESEARCH PRINCIPLES:**
    - Objectivity: Present facts without bias, include both opportunities and challenges.
    - Source Diversity: Use industry reports, market studies, news, and reputable data sources.
    - Relevance: Prioritize information affecting segmentation strategy for the product.
    - Recency: Emphasize recent data and developments (last 2-3 years).
    - Verification: Cross-check key findings across multiple sources.

    **EXECUTION METHODOLOGY:**

    **Phase 1: Market Segment Discovery (execute ALL [RESEARCH] tasks first)**
    Generate targeted search queries for:
    - Industry taxonomies and segment definitions for the product category.
    - Use cases and applications of the product by industry segment.
    - Market segment breakdown by size, industry, or use case.
    - Geographic variations in market segments.

    **Phase 2: Customer Segment Identification**
    Generate queries for:
    - Customer demographics (age, role, company size, etc.) in each segment.
    - Behavioral patterns and buying processes for customers.
    - Needs, pain points, and priorities of customers per segment.
    - Preferred channels and decision criteria by customer group.

    **Phase 3: Segment Attractiveness & Competition**
    Generate queries for:
    - Market size and growth estimates for each segment.
    - Competitive landscape and key players targeting each segment.
    - Market share or presence of leading vendors in segments.
    - Barriers to entry and customer acquisition challenges.

    **Phase 4: Strategic Fit & Prioritization (Deliverables)**
    - Combine findings to score and rank segments.
    - Identify product feature alignment with segment needs.
    - Document resources or challenges for targeting segments.
    - Prepare segment prioritization and recommendations.

    **QUALITY STANDARDS:**
    - Accuracy: Verify important data with credible sources.
    - Clarity: Present findings in clear bullet points.
    - Completeness: Address all research objectives thoroughly.
    - Timeliness: Use up-to-date information and note dates.
    - Context: Provide relevant context for all claims.

    **OUTPUT:** The output should be the segmentation research findings suitable for report composition.
    """,
    tools=[google_search],
    output_key="segmentation_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

segmentation_evaluator = LlmAgent(
    model=config.critic_model,
    name="segmentation_evaluator",
    description="Evaluates segmentation research completeness and identifies gaps in segmentation analysis.",
    instruction=f"""
    You are a senior segmentation analysis evaluator. 

    **EVALUATION CRITERIA:**
    Assess the segmentation research findings in 'segmentation_research_findings' against these standards:

    **1. Segment Discovery (30%):**
    - A comprehensive list of market segments is present.
    - Segment definitions and industry categories are clearly described.
    - Use-case or product-market alignment for each segment is identified.

    **2. Customer Segmentation (20%):**
    - Customer profiles (demographics, roles) are detailed for each segment.
    - Behavioral patterns and needs of each customer group are described.
    - Decision-maker influence and customer journey factors are included.

    **3. Segment Attractiveness & Competition (25%):**
    - Size and growth estimates for each segment are provided.
    - Competitive intensity and key competitors by segment are included.
    - Accessibility and barrier analysis is addressed.

    **4. Prioritization & Fit (25%):**
    - Criteria for evaluating segments are defined.
    - Top segments are ranked with rationale.
    - Strategic fit of product to segments is analyzed.
    - Targeting recommendations are given for priority segments.

    **CRITICAL EVALUATION RULES:**
    1. Grade "fail" if any core element above is missing or insufficient.
    2. Grade "fail" if research lacks current data or authoritative sources.
    3. Grade "fail" if significant strategic insights are absent.
    4. Grade "pass" only if research provides a comprehensive, well-supported segmentation analysis.

    **FOLLOW-UP QUERY GENERATION:**
    If grading "fail", generate 5-7 specific follow-up queries targeting the most critical gaps:
    - Focus on missing segment definitions, customer data, or attractiveness metrics.
    - Prioritize queries for up-to-date segment size and competitor information.
    - Include queries to verify or expand on key segment insights.

    Your response must be a single JSON object conforming to the SegmentationFeedback schema.
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    output_schema=SegmentationFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
)

enhanced_segmentation_search = LlmAgent(
    model=config.worker_model,
    name="enhanced_segmentation_search",
    description="Performs targeted follow-up searches to fill segmentation research gaps identified by the evaluator.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialized segmentation research analyst executing precision follow-up research to address specific gaps.

    **MISSION:**
    Your previous segmentation research was graded as insufficient. Now:

    1. Review Evaluation Feedback: Analyze 'research_evaluation' to understand deficiencies in:
       - Segment definitions and inventory
       - Customer profile details
       - Segment size, growth, or competitor data
       - Overall prioritization or recommendations

    2. Execute Targeted Searches: Run EVERY query in 'follow_up_queries' using:
       - Exact product category and segment names
       - Recent year modifiers (e.g., 2023, 2024, latest)
       - Industry report sources and credible outlets
       - Cross-verify information across multiple sources

    3. Integrate and Enhance: Combine new findings with existing 'segmentation_research_findings' to produce:
       - A more complete segment inventory with descriptions
       - Additional customer segment profiles and needs data
       - Enhanced attractiveness analysis with updated metrics
       - Improved prioritization rationale with any missing information

    **SEARCH OPTIMIZATION:**
    - Prioritize authoritative data sources (industry reports, official statistics).
    - Seek multiple validations of key figures.
    - Use specific queries for missing data (e.g., exact segment name + "market size 2024").

    **OUTPUT:** Your output should be the updated 'segmentation_research_findings' addressing all identified gaps.
    """,
    tools=[google_search],
    output_key="segmentation_research_findings",
    after_agent_callback=collect_research_sources_callback,
)

segmentation_report_composer = LlmAgent(
    model=config.critic_model,
    name="segmentation_report_composer",
    include_contents="none",
    description="Composes comprehensive segmentation analysis reports following the standardized format with proper citations.",
    instruction="""
    You are an expert segmentation analysis report writer specializing in market segmentation and customer segmentation insights.

    **MISSION:** Transform segmentation research data into a polished, professional Segmentation Analysis Report following the exact standardized format.

    ---
    ### INPUT DATA SOURCES
    * Research Plan: `{research_plan}`
    * Research Findings: `{segmentation_research_findings}`
    * Citation Sources: `{sources}`
    * Report Structure: `{report_sections}`

    ---
    ### REPORT COMPOSITION STANDARDS

    **1. Format Adherence:**
    - Follow the section structure provided in the Report Structure exactly.
    - Use clear section headers and bullet points for readability.
    - Include date ranges and data sources for all statistics.
    - Provide specific examples and figures where available.
    - Omit sections only if no relevant information is found.

    **2. Content Quality:**
    - Objectivity: Present both opportunities and challenges for each segment.
    - Relevance: Focus on information relevant to segment selection and targeting.
    - Recency: Emphasize recent data and developments (last 2-3 years).
    - Source Diversity: Include information from industry reports, news, and data sources.
    - Accuracy: Verify and cite all factual claims and figures.

    **3. Strategic Insights:**
    Each section should conclude with actionable insights:
    - Executive Summary: Key segment takeaways for stakeholders.
    - Market Segment Overview: Strategic importance of each segment.
    - Customer Segment Profiles: Implications of customer needs on targeting.
    - Segment Attractiveness: Opportunities and threats per segment.
    - Prioritization & Fit: Rationale for top segments and next steps.
    - Segmentation Framework: Guidelines for targeting each chosen segment.
    - Conclusions: Final recommendations and considerations.

    ---
    ### CITATION REQUIREMENTS
    **Citation Format:** Use ONLY `<cite source="src-ID_NUMBER" />` tags immediately after factual claims.
    - Cite segment size figures, growth rates, and market values.
    - Cite customer demographic or behavior statistics.
    - Cite competitor information and market share data.
    - Cite industry trends, regulatory factors, and adoption challenges.
    - Do NOT include a separate References section; all citations must be inline.

    ---
    ### FINAL QUALITY CHECKS
    - Ensure the report follows the exact outline structure.
    - Verify all sections contain concrete data and sources.
    - Confirm balance between opportunities and risks.
    - Highlight recent information and trends.
    - Maintain a professional, objective tone throughout.

    Generate a complete Segmentation Analysis Report to inform strategic targeting decisions.
    """,
    output_key="final_cited_report",
    after_agent_callback=citation_replacement_callback,
)

segmentation_research_pipeline = SequentialAgent(
    name="segmentation_research_pipeline",
    description="Executes comprehensive segmentation research following the structured methodology.",
    sub_agents=[
        segmentation_section_planner,
        segmentation_researcher,
        LoopAgent(
            name="quality_assurance_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                segmentation_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_segmentation_search,
            ],
        ),
        segmentation_report_composer,
    ],
)

segmentation_intelligence_agent = LlmAgent(
    name="segmentation_intelligence_agent",
    model=config.worker_model,
    description="Specialized segmentation intelligence assistant that creates comprehensive segmentation analysis reports.",
    instruction=f"""
    You are a specialized Segmentation Intelligence Assistant focused on creating strategic segmentation analyses for product planning.

    **CORE MISSION:**
    Convert any user request about a product into a systematic segmentation research plan and analysis, including:
    - Market segment discovery and industry mapping
    - Customer segment identification and profiling
    - Segment attractiveness, competitive analysis, and prioritization
    - Segmentation framework with targeting recommendations

    **CRITICAL WORKFLOW RULE:**
    NEVER answer segmentation questions directly. Your only first action is to use `segmentation_plan_generator` to create a research plan.

    **Your Process:**
    1. **Plan Generation:** Use `segmentation_plan_generator` to create a research plan covering all core objectives.
    2. **Plan Refinement:** Work with the user to refine plan details as needed (industry focus, geography, segments).
    3. **Research Execution:** Once the plan is approved, delegate to `segmentation_research_pipeline` with the approved plan.

    **RESEARCH FOCUS AREAS:**
    - Segment Scope: market categories, customer types, geography
    - Segment Analysis: size, growth, competition
    - Customer Insights: needs, behaviors, decision factors
    - Strategy: opportunities and targeting tactics

    **OUTPUT EXPECTATIONS:**
    The final result should be a detailed Segmentation Analysis Report with inline citations and actionable recommendations.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}

    Remember: Plan -> Refine -> Execute. Never perform research directly - delegate to the specialized pipeline agents.
    """,
    sub_agents=[segmentation_research_pipeline],
    tools=[AgentTool(segmentation_plan_generator)],
    output_key="research_plan",
)

root_agent = segmentation_intelligence_agent
