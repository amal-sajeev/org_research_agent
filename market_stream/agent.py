import json
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.models import Gemini
from google.genai import types as genai_types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.agent_tool import AgentTool
from .tools.mongoupload import update_project_report, create_blank_project
from .sub_agents.segmentation import segmentation_intelligence_agent
from .sub_agents.target_org_research import sales_intelligence_pipeline, sales_plan_generator
from .sub_agents.prospect_research import prospect_researcher
from .sub_agents.market_context import market_intelligence_agent
from .config import config

# ----------------------------------------------------------------------
# Storage Callback Functions
# ----------------------------------------------------------------------
def store_segmentation_report(callback_context: CallbackContext):
    """Store segmentation report after segmentation_intelligence_agent completes"""
    try:
        project_id = callback_context.state.get('project_id')
        project_id = project_id.replace('"','')
        segmentation_report = callback_context.state.get('segmentation_intelligence_agent')
        segmentation_html = callback_context.state.get("seg_html")

        if project_id and segmentation_report:
            update_project_report(
                project_id=project_id,
                report=segmentation_report,
                report_type="market_segment",
                html_report = segmentation_html
            )
            print(f"Segmentation report stored successfully for project {project_id}")
        else:
            print(f"Failed to store segmentation report - project_id: {project_id}, report exists: {bool(segmentation_report)}")
    except Exception as e:
        print(f"Error storing segmentation report: {e}")

def store_target_report(callback_context: CallbackContext):
    """Store Target intelligence report after sales_intelligence_agent completes"""
    try:
        project_id = callback_context.state.get('project_id')
        project_id = project_id.replace('"','')
        target_report = callback_context.state.get('sales_intelligence_agent')
        target_html = callback_context.state.get("target_html")
        
        if project_id and target_report:
            update_project_report(
                project_id=project_id,
                report=target_report,
                report_type="target_org_research",
                html_report = target_html
            )
            print(f"Target intelligence report stored successfully for project {project_id}")
        else:
            print(f"Failed to store org report - project_id: {project_id}, report exists: {bool(org_report)}")
    except Exception as e:
        print(f"Error storing organizational intelligence report: {e}")

def store_prospect_report(callback_context: CallbackContext):
    """Store prospect research report after prospect_researcher completes"""
    try:
        project_id = callback_context.state.get('project_id')
        project_id = project_id.replace('"','')
        prospect_report = callback_context.state.get('prospect_researcher')
        
        if project_id and prospect_report:
            update_project_report(
                project_id=project_id,
                report=prospect_report,
                report_type="prospect_research"
            )
            
            print("###################################################################################")
            print("###################################################################################")
            print(f"Prospect research report stored successfully for project {project_id}")
            print("###################################################################################")
            print("###################################################################################")
        else:
            print(f"Failed to store prospect report - project_id: {project_id}, report exists: {bool(prospect_report)}")
    except Exception as e:
        print(f"Error storing prospect research report: {e}")

def store_context_report(callback_context: CallbackContext):
    """Store prospect research report after prospect_researcher completes"""
    try:
        project_id = callback_context.state.get('project_id')
        project_id = project_id.replace('"','')
        context_report = callback_context.state.get('market_intelligence_agent')
        context_html = callback_context.state.get("context_html")

        if project_id and context_report:
            update_project_report(
                project_id=project_id,
                report=context_report,
                report_type="market_context",
                html_report= context_html
            )
            print(f"Market context report stored successfully for project {project_id}")
        else:
            print(f"Failed to store Market context report - project_id: {project_id}, report exists: {bool(context_report)}")
    except Exception as e:
        print(f"Error storing Market context report: {e}")



def extract_project_id(callback_context: CallbackContext):
    """Extract and store project_id from initial input for use by storage callbacks"""
    try:
        # Get the original input and extract project_id
        input_data = callback_context.input_data
        if isinstance(input_data, dict) and 'project_id' in input_data:
            callback_context.state['project_id'] = input_data['project_id']
        elif isinstance(input_data, str):
            # Try to parse as JSON if it's a string
            try:
                parsed = json.loads(input_data)
                if 'project_id' in parsed:
                    callback_context.state['project_id'] = parsed['project_id']
            except:
                # If not JSON, try to extract project_id from text
                # This is a fallback - adjust based on your input format
                pass
        
        print(f"Project ID extracted: {callback_context.state.get('project_id')}")
    except Exception as e:
        print(f"Error extracting project_id: {e}")

# ----------------------------------------------------------------------
# Ensure output_key is consistent for all imported sub-agents
# ----------------------------------------------------------------------

segmentation_intelligence_agent.output_key = "segmentation_intelligence_agent"
# prospect_researcher.output_key = "prospect_researcher"

# Add after-agent callbacks for storage
segmentation_intelligence_agent.after_agent_callback = [store_segmentation_report]
# organizational_intelligence_agent.after_agent_callback = [store_organizational_report]
market_intelligence_agent.after_agent_callback = [store_context_report]
prospect_researcher.after_agent_callback = [store_prospect_report]

# ----------------------------------------------------------------------
# Input Analyzer (simplified - just extracts project_id)
# ----------------------------------------------------------------------
input_analyzer = LlmAgent(
    name="input_analyzer",
    model = config.worker_model,
    description="Analyzes user input and extracts project information.",
    instruction="""
        Analyze the user input and extract any relevant information for the intelligence pipeline.
        
        Output ONLY a JSON object:
        {
            "processed": true,
            "input_summary": "Brief summary of the user request"
        }
    """,
    output_key="input_analysis",
    after_agent_callback=[extract_project_id]
)

# ----------------------------------------------------------------------
# Prompt Builders
# ----------------------------------------------------------------------
segmentation_prompt_builder = LlmAgent(
    name="segmentation_prompt_builder",
    model = config.worker_model,
    description="Generates JSON input for segmentation_intelligence_agent from user input.",
    instruction="""
        Based on the user input and the market context report, create a JSON object for market segmentation analysis. ONLY USE MARKET CONTEXT REPORT FOR FILLING THE industry_market AND current_understanding FIELDS.
        
        market context report: {market_intelligence_agent}
        Output ONLY a valid JSON object in the following format. Do not include any extra text or commentary.
        {
            "product_name": "Your Product Name Here",
            "product_description": "A detailed description of your product or service and what it does.",
            "industry_market": "The industry or market in which your product operates",
            "current_understanding": "Any existing insights about target demographics, market challenges, or strategic context from the user input."
        }
        
        Extract the product/service information from the user input.
        Be specific and detailed in the product_description based on user input.
    """,
    output_key="segmentation_agent_input"
)

org_prompt_builder = LlmAgent(
    name="org_prompt_builder",
    model = config.worker_model,
    description="Generates JSON input for organizational_intelligence_agent using only user input.",
    instruction="""
        Based solely on the user input, create a JSON object for organizational intelligence research.
        
        Output ONLY a valid JSON object in the following format. Do not include any extra text or commentary.
        {
            "companies": [
                {
                    "name": "Company Name",
                    "ticker_symbol": "TICK",
                    "industry": "Industry mentioned or inferred from user input",
                    "alternate_name": "Alternative company name if applicable",
                    "website": "company website if known"
                }
            ],
            "products": [
                {
                    "product_or_service": "Product name from user input",
                    "product_type": "Product category inferred from user input",
                    "product_link": ""
                }
            ]
        }
        
        Extract all company and product information directly from the user input.
        Infer industry and product categories based on what the user has described.
    """,
    output_key="org_agent_input"
)

market_context_prompt_builder = LlmAgent(
    name="market_context_prompt_builder",
    model = config.worker_model,
    description="Generates JSON input for market_intelligence_agent using only user_input.",
    instruction="""
        Using user input, create a JSON object as the input prompt for market context research.
        Output ONLY a valid JSON object:
        {
            "companies": [
                {
                    "name": "Client organization Name",
                    "ticker_symbol": "TICK",
                    "industry": "Industry mentioned or inferred from user input",
                    "alternate_name": "Client organization Name if applicable",
                    "website": "company website if known"
                }
            ],
            "products": [
                {
                    "product_or_service": "Product name from user input",
                    "product_type": "Product category inferred from user input",
                    "product_link": ""
                }
            ]
        }
        
        Incorporate insights from both the segmentation and organizational intelligence reports.
    """,
    output_key="market_context_input"
)


prospect_prompt_builder = LlmAgent(
    name="prospect_prompt_builder",
    model = config.worker_model,
    description="Generates JSON input for prospect_researcher using segmentation and organizational reports.",
    instruction="""
        Using the user input, segmentation report, and organizational report, create a JSON object for prospect research.
        
        Segmentation Report: {segmentation_intelligence_agent}
        
        Output ONLY a valid JSON object:
        {
            "products_services": [
                {
                    "name": "Product name from segmentation analysis",
                    "description": "Comprehensive description incorporating segmentation and organizational insights"
                }
            ],
            "company_information": "Background about the company based on organizational intelligence",
            "known_competitors": ["List of competitors from organizational intelligence"],
            "target_context": "Additional context from segmentation and organizational analysis"
        }
        
        Incorporate insights from both the segmentation and organizational intelligence reports.
    """,
    output_key="prospect_agent_input"
)


# ----------------------------------------------------------------------
# CONDITIONAL SALES INTELLIGENCE AGENTS
# ----------------------------------------------------------------------
conditional_sales_prompt_builder = LlmAgent(
    name="conditional_sales_prompt_builder",
    model = config.worker_model,
    description="Conditionally generates JSON input for sales_intelligence_agent or passes through empty result.",
    instruction="""
        Check the sales activator: {sales_activator}
        
        If sales_activator.activate_sales is TRUE, then create a JSON object for sales intelligence research using the user input and previous reports.
        
        If sales_activator.activate_sales is FALSE, output exactly: {{"skip_sales": true}}
        
        When creating sales intelligence input (if activate_sales is TRUE):
        
        Market Intelligence Report: {market_intelligence_agent}
        
        Output a valid JSON object:
        {
            "products": [
                {
                    "name": "Product name from previous reports",
                    "category": "Product category from segmentation analysis",
                    "details": "Enhanced product details from organizational intelligence"
                }
            ],
            "target_organizations": [
                {
                    "name": "Organization 1 name from sales_activator.organizations_mentioned",
                    "industry": "Industry from market analysis",
                    "context": "Context about why this organization is a target"
                },
                {
                    "name": "Organization 2 name from sales_activator.organizations_mentioned",
                    "industry": "Industry from market analysis",
                    "context": "Context about why this organization is a target"
                }
            ],
            "research_objectives": "Research objectives focused on the specific organizations mentioned by the user"
        }
        
        Use the organizations_mentioned from sales_activator.organizations_mentioned as the primary targets.
    """,
    output_key="sales_agent_input"
)

conditional_sales_intelligence_agent = LlmAgent(
    name="conditional_sales_intelligence_agent",
    model = config.worker_model,
    description="Conditionally executes sales intelligence research or skips if no specific targets identified.",
    instruction="""
        Check the sales_agent_input: {sales_agent_input}
        
        If sales_agent_input contains "skip_sales": true, then output exactly:
        {{"skipped": true, "reason": "No specific target organizations identified in user input"}}
        
        Otherwise, You are a specialized Sales Intelligence Assistant focused on comprehensive product-organization fit analysis for account-based selling and strategic sales planning.

    **CORE MISSION:**
    Convert ANY user request about products and target organizations into a systematic research plan that generates actionable sales intelligence through mandatory 2-step execution.

    **CRITICAL WORKFLOW RULE - MANDATORY 2-STEP PROCESS:**
    1. FIRST: Use `sales_plan_generator` to create a research plan
    2. SECOND: IMMEDIATELY call the `sales_intelligence_pipeline` sub-agent with the generated plan
    
    You MUST complete BOTH steps in sequence. Never stop after step 1.

    **INPUT PROCESSING:**
    You will receive requests in various formats:
    - "Research [Company A, Company B] for selling [Product X, Product Y]"
    - "Analyze fit between our [Product] and [Organization]"
    - "Sales intelligence for [Products] targeting [Organizations]"
    - Lists of companies and products in any combination

    **MANDATORY EXECUTION SEQUENCE:**
    
    **STEP 1 - Plan Generation (REQUIRED):**
    Use `sales_plan_generator` to create a 5-phase research plan covering:
    - Product Intelligence (competitive landscape, value props, customer success)
    - Organization Intelligence (structure, priorities, decision-makers)
    - Technology & Vendor Landscape (current solutions, gaps, procurement)
    - Stakeholder Mapping (decision-makers, influencers, champions)
    - Competitive & Risk Assessment (threats, obstacles, timing)

    **STEP 2 - Research Execution (MANDATORY FOLLOW-UP):**
    IMMEDIATELY after receiving the plan from step 1, you MUST invoke the `sales_intelligence_pipeline` sub-agent.
    Pass the generated research plan to `sales_intelligence_pipeline` for complete execution.
    
    **AUTOMATIC EXECUTION REQUIREMENT:**
    - You will proceed with BOTH steps without asking for approval
    - You will NOT stop after plan generation
    - You will NOT wait for user confirmation between steps
    - The process is: Plan → Execute → Deliver (all automatic)

    **RESEARCH FOCUS AREAS:**
    - **Product Analysis:** Competitive positioning, value propositions, customer success metrics
    - **Organization Analysis:** Decision-makers, business priorities, technology gaps
    - **Fit Assessment:** Product-organization compatibility matrices and opportunity scoring  
    - **Competitive Intelligence:** Incumbent solutions, vendor relationships, competitive threats
    - **Sales Strategy:** Stakeholder engagement plans, messaging themes, timing considerations
    - **Risk Assessment:** Budget constraints, competitive entrenchment, cultural fit challenges

    **FINAL OUTPUT REQUIREMENT:**
    The completed execution must produce a comprehensive Sales Intelligence Report with 9 standardized sections:
    1. Executive Summary (opportunities, risks, priorities)
    2. Product Overview(s) (value props, differentiators, use cases)
    3. Target Organization Profiles (structure, priorities, vendor landscape)
    4. Product–Organization Fit Analysis (cross-matrix with scores)
    5. Competitive Landscape (per organization analysis)
    6. Stakeholder Engagement Strategy (who, how, when)
    7. Risks & Red Flags (obstacles and mitigation)
    8. Next Steps & Action Plan (immediate, medium, long-term)
    9. Appendices (detailed profiles, contacts, references)

    **WORKFLOW VALIDATION:**
    Before finishing, confirm you have:
    ✓ Generated a research plan using sales_plan_generator
    ✓ Executed the research using sales_intelligence_pipeline
    ✓ Delivered a complete Sales Intelligence Report
    
    If any step is missing, complete it before responding to the user.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}

    Remember: This is a 2-step mandatory process. Plan generation alone is incomplete - you MUST also execute the research pipeline.
    """,
    sub_agents=[sales_intelligence_pipeline],
    tools=[AgentTool(sales_plan_generator)],
    output_key="sales_research_plan",
    after_agent_callback = [store_target_report]
)

# ----------------------------------------------------------------------
# Project Creator Agent
# ----------------------------------------------------------------------
project_creator = LlmAgent(
    name="project_creator",
    model = config.worker_model,
    description="Creates a blank MongoDB project document using the provided project ID.",
    instruction="""
        You will receive input that contains a project_id. Extract the project_id and use the create_blank_project tool to create a new blank project document in MongoDB.
        
        After successfully creating the project, respond with just the project_id as a string and nothing else.
    """,
    tools=[create_blank_project],
    output_key="project_id"
)

determine_sales = LlmAgent(
    name = "determine_sales",
    model = config.worker_model,
    description = "Checks user input and determines if target_intelligence agent is needed.",
    instruction= """
        Analyze the user input to determine if they have mentioned specific organizations/companies they want to target.
        
        Look for:
        - Direct mentions of target companies (e.g., "we want to target IBM and Oracle")
        - Named organizations they want to sell to
        
        Do NOT consider these as specific organizations:
        - General industry references (e.g., "tech companies", "healthcare organizations")
        - Market segments (e.g., "enterprise customers", "small businesses") 
        - Geographic references (e.g., "companies in Silicon Valley")
        - Company types without names (e.g., "SaaS companies", "manufacturing firms")
        - Size descriptors (e.g., "Fortune 500", "startups", "mid-market")
        
        Output ONLY a JSON object:
            {
                "activate_sales": TRUE if target companies are mentioned or FALSE if target companies is an empty list or not mentioned at all,
                "organizations_mentioned": ["Company 1", "Company 2", etc.]
            }
        TRUE if target companies are mentioned
        FALSE if target companies are an empty list or not mentioned at all
        """,
    output_key="sales_activator"
)

# ----------------------------------------------------------------------
# Simplified Sequential Agent
# ----------------------------------------------------------------------
simplified_intelligence_agent = SequentialAgent(
    name="simplified_intelligence_agent",
    description="""
        Runs a intelligence analysis pipeline with automatic storage:
        
        1. Analyze user input and extract project information
        2. Create blank project document in MongoDB
        3. Market Segmentation Analysis → Auto-stored to market_segment via callback
        4. Organizational Intelligence Research → Auto-stored to client_org_research via callback
        5. Prospect Research → Auto-stored to prospect_research via callback
        
        Each research agent has an after_agent_callback that automatically stores its report to MongoDB
        using the project_id extracted from the initial input.
    """,
    sub_agents=[
        input_analyzer,                         # Analyze input + extract project_id
        project_creator,                        #ms Create blank project in MongoDB
        determine_sales,
        market_context_prompt_builder,
        market_intelligence_agent,
        segmentation_prompt_builder,            # Build segmentation prompt
        segmentation_intelligence_agent,        # Execute segmentation + auto-store
        conditional_sales_prompt_builder,
        conditional_sales_intelligence_agent,
        prospect_prompt_builder,                # Build prospect prompt
        prospect_researcher,                    # Execute prospect research + auto-store
    ]
)

root_agent = simplified_intelligence_agent