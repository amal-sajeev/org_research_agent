TARGET_TEMPLATE = """
    You are an expert Target Organization Research HTML report generator. You were given a fixed HTML template (do not alter it) that contains bracketed placeholders like [[PRODUCT_NAME]], [[REGIONAL_TABLE_JSON]], etc. Your job: **output only one artifact — the complete HTML file** with every bracketed placeholder replaced according to the rules below. Do not output commentary, analysis, questions, or any extra text.
    
    INPUT
    - `sales_research_findings` (text / bullet list) — use facts from here to populate placeholders.
    - `citations` (list of objects: {id, title, url, accessed}) — map these to reference anchors in the References section.
    - `sales_intelligence_agent` (report-structure instructions) — follow if relevant.

    RETURN
    - Exactly one file: the completed HTML document string. No other output allowed.
    ---
    ### INPUT DATA SOURCES
    * Research Findings: {sales_research_findings}
    * Citation Sources: {citations}
    * Report Structure: {sales_intelligence_agent}

    ---
    Global Rules
    - Do **not** invent facts. Map only what exists in the Markdown.
    - Respect all existing HTML comments inside the template (they are implementation guidance). Keep comments in the output unless a comment explicitly says to remove it.
    - Keep all `<section>` IDs, class names, and markup unchanged; only replace placeholders.
    - Fill `[[Company]]`, `[[City]]`, `[[Team]]` from the Markdown (front matter or first mention). If missing, set:
    - `[[Company]] = [[MISSING_COMPANY]]`
    - `[[City]] = [[MISSING_CITY]]`
    - `[[Team]] = [[MISSING_TEAM]]`
    - Replace every `[[...]]` placeholder with concise content extracted from the matching Markdown section. If the corresponding content is absent, leave the placeholder as `[[MISSING_<NAME>]]`.
    - Maintain bullet/numbered lists as lists; keep tables as tables; keep short, action-oriented phrasing.
    - Inline citation tags in the Markdown (e.g., `[12]`, `(ref: 12)`) should be preserved verbatim in the relevant sentences. The “References”/“Citations” section should render items `[n]` with `SOURCE_NAME` and URL if provided in Markdown. Do **not** fabricate URLs.
    - Do **not** attach scripts, external CSS, or images. No links to assets other than those in the Markdown references.
    - Escape any raw Markdown artifacts that could break HTML.
    ### HTML TEMPLATE
    Use this EXACT template structure, replace the bracketed placeholders and create the tables and charts with actual data:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Intelligence Report - [[PRODUCT NAME]]</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #2980b9;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --text-color: #333;
            --border-radius: 8px;
            --box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: var(--text-color);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: var(--box-shadow);
        }
        
        .report-header {
            text-align: center;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            color: var(--primary-color);
            font-size: 2.5em;
            margin: 0;
            font-weight: 700;
        }
        
        .report-subtitle {
            font-size: 1.2em;
            color: #7f8c8d;
            margin-top: 10px;
        }
        
        .report-meta {
            background: var(--light-color);
            padding: 15px;
            border-radius: var(--border-radius);
            margin: 20px 0;
            border-left: 4px solid var(--secondary-color);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        h2 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 1.8em;
        }
        
        h3 {
            color: #34495e;
            margin-top: 30px;
            font-size: 1.4em;
            border-left: 4px solid var(--secondary-color);
            padding-left: 15px;
        }
        
        h4 {
            color: var(--primary-color);
            margin-top: 20px;
            font-size: 1.2em;
        }
        
        /* Citation styling - Wikipedia-style numbered hyperlinks */
        .citation-link {
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: bold;
            font-size: 0.9em;
            vertical-align: super;
        }
        
        .citation-link:hover {
            text-decoration: underline;
        }
        
        .toc-list {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
            columns: 2;
            column-gap: 30px;
        }
        
        .toc-list li {
            margin: 8px 0;
            break-inside: avoid;
        }
        
        .toc-list a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
            display: block;
            padding: 5px 0;
            transition: all 0.3s ease;
        }
        
        .toc-list a:hover {
            color: var(--secondary-color);
            text-decoration: underline;
            padding-left: 5px;
        }
        
        .data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .data-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .data-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--box-shadow);
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--primary-color);
            margin: 10px 0;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .section-highlight {
            background: #e8f6ff;
            padding: 20px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--secondary-color);
            margin: 20px 0;
        }
        
        .risk-warning {
            background: #fff5f5;
            padding: 20px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--danger-color);
            margin: 20px 0;
        }
        
        .key-insights {
            background: #f0fff4;
            padding: 20px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--success-color);
            margin: 20px 0;
        }
        
        .content-section {
            margin: 25px 0;
        }
        
        .sub-section {
            margin: 20px 0;
            padding-left: 20px;
            border-left: 2px solid #e9ecef;
        }
        
        .bullet-points {
            padding-left: 20px;
        }
        
        .bullet-points li {
            margin: 8px 0;
            line-height: 1.5;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
        }
        
        .data-table th, .data-table td {
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }
        
        .data-table th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }
        
        .data-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .recommendation-box {
            background: #e8f5e8;
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid #d4edda;
            margin: 15px 0;
        }
        
        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
        }
        
        .tag-high {
            background-color: #ffeaea;
            color: var(--danger-color);
        }
        
        .tag-medium {
            background-color: #fff4e0;
            color: var(--warning-color);
        }
        
        .tag-low {
            background-color: #e8f6ff;
            color: var(--secondary-color);
        }
        
        .key-findings {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .finding-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-top: 4px solid var(--secondary-color);
        }
        
        .finding-card.critical {
            border-top-color: var(--danger-color);
        }
        
        .finding-card.opportunity {
            border-top-color: var(--success-color);
        }
        
        .progress-bar {
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--secondary-color);
            border-radius: 4px;
        }
        
        /* Market Segment specific styling */
        .segment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .segment-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-top: 4px solid var(--secondary-color);
            transition: transform 0.3s ease;
        }
        
        .segment-card:hover {
            transform: translateY(-5px);
        }

        /* Additional styles for the report */
        .executive-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: var(--border-radius);
            margin: 30px 0;
        }

        .executive-summary h2 {
            color: white;
            border-bottom: 2px solid rgba(255,255,255,0.3);
        }

        .stakeholder-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .stakeholder-table th, .stakeholder-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        .stakeholder-table th {
            background-color: var(--light-color);
            font-weight: 600;
        }

        .fit-matrix {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }

        .fit-score {
            text-align: center;
            padding: 15px;
            border-radius: var(--border-radius);
            font-weight: bold;
        }

        .fit-high { background: #d4edda; color: var(--success-color); }
        .fit-medium { background: #fff3cd; color: var(--warning-color); }
        .fit-low { background: #f8d7da; color: var(--danger-color); }

        .risk-matrix {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .risk-item {
            padding: 15px;
            border-radius: var(--border-radius);
            border-left: 4px solid;
        }

        .risk-high { 
            background: #fff5f5; 
            border-left-color: var(--danger-color);
        }

        .risk-medium { 
            background: #fff9e6; 
            border-left-color: var(--warning-color);
        }

        .competitive-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .competitor-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
        }

        .next-steps-timeline {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            margin: 20px 0;
        }

        .timeline-item {
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .timeline-badge {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--secondary-color);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
        }

        .appendix-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .appendix-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            border-top: 3px solid var(--secondary-color);
        }

        .citation-footer {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            margin-top: 40px;
            font-size: 0.9em;
        }

        .citation-item {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid var(--secondary-color);
            padding-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="report-header">
            <!-- AI: Use a concise, professional title for the report. -->
            <h1>[[REPORT_TITLE]]</h1>

            <!-- AI: Subtitle should specify the product/solution and client context (e.g., "Master AI: The Leitner Way (Global Knowledge Tech)"). -->
            <div class="report-subtitle">[[REPORT_SUBTITLE]]</div>
        </div>

        <div class="report-meta">
            <!-- AI: Insert project ID or tracking code for internal reference. -->
            <div><strong>Project ID:</strong> [[PROJECT_ID]]</div>

            <!-- AI: Specify the exact client account name and location. -->
            <div><strong>Target Account:</strong> [[TARGET_ACCOUNT]]</div>

            <!-- AI: Identify the AI agent, researcher, or system generating the report. -->
            <div><strong>Prepared by:</strong> [[AUTHOR]]</div>

            <!-- AI: Insert the report date. Use "Current Analysis" if generated live. -->
            <div><strong>Date:</strong> [[REPORT_DATE]]</div>
        </div>

        <div class="section-highlight">
            <!-- AI: Always include a note on how citations and references are handled. 
                 - State whether citations are inline, endnotes, or hyperlinks. 
                 - Mention what happens if public/verified data is not available. -->
            <p><strong>Note on citations:</strong> [[CITATION_NOTE]]</p>
        </div>

        <!-- EXECUTIVE SUMMARY -->
        <section id="executive-summary" class="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="data-grid">
                <!-- AI: Fill metrics with quantified insights about the client.
                     - Use numbers with clear units (e.g., M+, K+, $, weeks).
                     - Each label describes the metric context. -->
                <div class="data-card">
                    <div class="metric-value">[[METRIC_VALUE_1]]</div>
                    <div class="metric-label">[[METRIC_LABEL_1]]</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[METRIC_VALUE_2]]</div>
                    <div class="metric-label">[[METRIC_LABEL_2]]</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[METRIC_VALUE_3]]</div>
                    <div class="metric-label">[[METRIC_LABEL_3]]</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[METRIC_VALUE_4]]</div>
                    <div class="metric-label">[[METRIC_LABEL_4]]</div>
                </div>
            </div>

            <div class="key-insights">
                <h4>Key Opportunity</h4>
                <!-- AI: Summarize the single biggest client opportunity. 
                     - Keep it 1–2 sentences.
                     - Link it to AI adoption, L&D investment, or strategic goals. -->
                <p>[[KEY_OPPORTUNITY]]</p>
            </div>
        </section>
      
      	<div class="content-section">
                <h4>Strategic Goals</h4>
                <ul class="bullet-points">
                    <!-- AI: List client-specific goals with clear time horizons.
                         - Short-term: immediate, 0–3 months, usually pilots or proofs-of-concept.
                         - Mid-term: 3–12 months, typically scaling or converting pilots into broader adoption. -->
                    <li><strong>Short-term (0–3 months):</strong> [[STRATEGIC_GOAL_SHORT_TERM]]</li>
                    <li><strong>Mid-term (3–12 months):</strong> [[STRATEGIC_GOAL_MID_TERM]]</li>
                </ul>
            </div>

        <!-- Product Intelligence -->
        <section id="product-intelligence">
            <h2>Product Intelligence: [[PRODUCT_NAME]]</h2>
            
            <div class="section-highlight">
                <h4>Product Definition</h4>
                <!-- AI: Provide a clear, concise product definition.
                     - Focus on the enterprise use-case.
                     - Mention delivery format (e.g., SaaS, tactile kit, blended).
                     - Highlight any differentiating features like compliance-readiness or analytics. -->
                <p>[[PRODUCT_DEFINITION]]</p>
            </div>

            <div class="key-findings">
                <div class="finding-card opportunity">
                    <h4>Core Value Propositions</h4>
                    <!-- AI: List 3–5 key value propositions.
                         - Each should highlight measurable or strategic client benefit.
                         - Format: <strong>[VALUE_TAG]:</strong> [VALUE_DESCRIPTION]. -->
                    <ul class="bullet-points">
                        <li><strong>[[VALUE_TAG_1]]:</strong> [[VALUE_DESCRIPTION_1]]</li>
                        <li><strong>[[VALUE_TAG_2]]:</strong> [[VALUE_DESCRIPTION_2]]</li>
                        <li><strong>[[VALUE_TAG_3]]:</strong> [[VALUE_DESCRIPTION_3]]</li>
                    </ul>
                </div>

                <div class="finding-card">
                    <h4>Technical Integration</h4>
                    <!-- AI: Provide 3–4 technical integration features.
                         - Focus on interoperability with enterprise systems (LMS, SSO, analytics).
                         - Each item should be short, specific, and technical. -->
                    <ul class="bullet-points">
                        <li>[[TECH_INTEGRATION_1]]</li>
                        <li>[[TECH_INTEGRATION_2]]</li>
                        <li>[[TECH_INTEGRATION_3]]</li>
                    </ul>
                </div>

                <div class="finding-card critical">
                    <h4>Evidence Gap</h4>
                    <!-- AI: Identify one or more evidence gaps or missing validation.
                         - Highlight where proof is lacking (e.g., peer-reviewed studies, enterprise pilots).
                         - Keep it factual and neutral. -->
                    <p>[[EVIDENCE_GAP]]</p>
                </div>
            </div>
        </section>

        <!-- Organization Intelligence -->
        <section id="organization-intelligence">
            <h2>Organization Intelligence: [[ORG_NAME]] ([[ORG_LOCATION]])</h2>
            
            <div class="segment-grid">
                <div class="segment-card">
                    <h4>Company Fundamentals</h4>
                    <!-- AI: Provide a concise description of the company.
                         - Cover headquarters, industry, size, and relevance to L&D/AI adoption.
                         - 1–2 sentences only. -->
                    <p>[[ORG_DESCRIPTION]]</p>

                    <div class="data-grid">
                        <!-- AI: Insert key company training/learning stats. 
                             - Each metric must have a numeric value + a clear label (FY-based if available).
                             - Use M+ / K+ notation for large figures. -->
                        <div class="data-card">
                            <div class="metric-value">[[ORG_METRIC_VALUE_1]]</div>
                            <div class="metric-label">[[ORG_METRIC_LABEL_1]]</div>
                        </div>
                        <div class="data-card">
                            <div class="metric-value">[[ORG_METRIC_VALUE_2]]</div>
                            <div class="metric-label">[[ORG_METRIC_LABEL_2]]</div>
                        </div>
                    </div>
                </div>
                
                <div class="segment-card">
                    <h4>Technology Stack</h4>
                    <!-- AI: List 3–4 key enterprise systems/tools relevant to learning and identity management.
                         - Format: <strong>[SYSTEM_TYPE]:</strong> [SYSTEM_NAME or DESCRIPTION].
                         - Keep phrasing consistent and professional. -->
                    <ul class="bullet-points">
                        <li><strong>[[TECH_STACK_TYPE_1]]:</strong> [[TECH_STACK_DETAIL_1]]</li>
                        <li><strong>[[TECH_STACK_TYPE_2]]:</strong> [[TECH_STACK_DETAIL_2]]</li>
                        <li><strong>[[TECH_STACK_TYPE_3]]:</strong> [[TECH_STACK_DETAIL_3]]</li>
                    </ul>
                </div>
                
                <div class="segment-card">
                    <h4>Strategic Themes</h4>
                    <!-- AI: Provide 3–5 strategic priorities/themes guiding the organization.
                         - Keep each item short (max 6–7 words).
                         - Focus on AI, digital transformation, workforce learning, or partnerships. -->
                    <ul class="bullet-points">
                        <li>[[STRATEGIC_THEME_1]]</li>
                        <li>[[STRATEGIC_THEME_2]]</li>
                        <li>[[STRATEGIC_THEME_3]]</li>
                        <li>[[STRATEGIC_THEME_4]]</li>
                    </ul>
                </div>
            </div>
        </section>


        <!-- Product-Organization Fit Analysis -->
        <section id="fit-analysis">
            <h2>Product ↔ Organization Fit Analysis</h2>
            
            <div class="section-highlight">
                <h4>Strategic Alignment Assessment</h4>
                <!-- AI: Provide a 2–3 sentence assessment.
                     - Explain why the product is relevant for the target organization.
                     - Mention alignment with current initiatives (e.g., AI skilling, digital transformation).
                     - End with a probability rating in bold (Low, Medium, Medium-High, High). -->
                <p>[[STRATEGIC_ALIGNMENT_ASSESSMENT]] <strong>Probability: [[ALIGNMENT_PROBABILITY]]</strong></p>
            </div>

            <h3>Fit Matrix Analysis</h3>
            <div class="fit-matrix">
                <!-- AI: Provide fit scores for 3–4 categories.
                     - Categories should be client-relevant (e.g., Strategic Alignment, Technical Integration, Procurement).
                     - Score values must be standardized (LOW, MEDIUM, MEDIUM-HIGH, HIGH). -->
                <div class="fit-score fit-[[FIT_LEVEL_1]]">
                    <div>[[FIT_CATEGORY_1]]</div>
                    <div style="font-size: 1.5em; margin-top: 10px;">[[FIT_SCORE_1]]</div>
                </div>
                <div class="fit-score fit-[[FIT_LEVEL_2]]">
                    <div>[[FIT_CATEGORY_2]]</div>
                    <div style="font-size: 1.5em; margin-top: 10px;">[[FIT_SCORE_2]]</div>
                </div>
                <div class="fit-score fit-[[FIT_LEVEL_3]]">
                    <div>[[FIT_CATEGORY_3]]</div>
                    <div style="font-size: 1.5em; margin-top: 10px;">[[FIT_SCORE_3]]</div>
                </div>
            </div>

            <div class="recommendation-box">
                <h4>Recommended Pilot Profile</h4>
                <!-- AI: Suggest a pilot configuration.
                     - Sponsor: Role/title most likely to own the pilot.
                     - Duration: Suggested pilot length (weeks/months).
                     - Cohort Size: Learner group size range.
                     - Success Target: Measurable performance/retention uplift. -->
                <table class="stakeholder-table">
                    <tr>
                        <td><strong>Sponsor:</strong></td>
                        <td>[[PILOT_SPONSOR]]</td>
                    </tr>
                    <tr>
                        <td><strong>Duration:</strong></td>
                        <td>[[PILOT_DURATION]]</td>
                    </tr>
                    <tr>
                        <td><strong>Cohort Size:</strong></td>
                        <td>[[PILOT_COHORT_SIZE]]</td>
                    </tr>
                    <tr>
                        <td><strong>Success Target:</strong></td>
                        <td>[[PILOT_SUCCESS_TARGET]]</td>
                    </tr>
                </table>
            </div>
        </section>


         <!-- Competitive Landscape -->
        <section id="competitive-landscape">
            <h2>Competitive Landscape & Positioning</h2>
            
            <div class="competitive-grid">
                <!-- AI: For each competitor group:
                     - Heading = Competitor category/type.
                     - Paragraph = Key players or platforms (comma-separated).
                     - Tag = Competitive strength (LOW, MEDIUM, HIGH) with a short label. -->
                <div class="competitor-card">
                    <h4>[[COMPETITOR_CATEGORY_1]]</h4>
                    <p>[[COMPETITOR_LIST_1]]</p>
                    <div class="tag tag-[[COMPETITOR_TAG_LEVEL_1]]">[[COMPETITOR_TAG_LABEL_1]]</div>
                </div>
                <div class="competitor-card">
                    <h4>[[COMPETITOR_CATEGORY_2]]</h4>
                    <p>[[COMPETITOR_LIST_2]]</p>
                    <div class="tag tag-[[COMPETITOR_TAG_LEVEL_2]]">[[COMPETITOR_TAG_LABEL_2]]</div>
                </div>
                <div class="competitor-card">
                    <h4>[[COMPETITOR_CATEGORY_3]]</h4>
                    <p>[[COMPETITOR_LIST_3]]</p>
                    <div class="tag tag-[[COMPETITOR_TAG_LEVEL_3]]">[[COMPETITOR_TAG_LABEL_3]]</div>
                </div>
                <div class="competitor-card">
                    <h4>[[DIFFERENTIATION_HEADING]]</h4>
                    <p>[[DIFFERENTIATION_DESCRIPTION]]</p>
                    <div class="tag tag-[[DIFFERENTIATION_TAG_LEVEL]]">[[DIFFERENTIATION_TAG_LABEL]]</div>
                </div>
            </div>

            <h3>Objection Handling Playbook</h3>
            <!-- AI: Populate 3–5 common objections and corresponding rebuttal strategies.
                 - Objection: Phrase it as if spoken by client (use quotes).
                 - Rebuttal Strategy: Concise, actionable counter-strategy with measurable or practical detail. -->
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Objection</th>
                        <th>Rebuttal Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>[[OBJECTION_1]]</td>
                        <td>[[REBUTTAL_1]]</td>
                    </tr>
                    <tr>
                        <td>[[OBJECTION_2]]</td>
                        <td>[[REBUTTAL_2]]</td>
                    </tr>
                    <tr>
                        <td>[[OBJECTION_3]]</td>
                        <td>[[REBUTTAL_3]]</td>
                    </tr>
                </tbody>
            </table>
        </section>


        <!-- Stakeholder Mapping -->
        <section id="stakeholder-mapping">
            <h2>Stakeholder Mapping & Procurement Path</h2>
            
            <h3>Target Stakeholders (Priority Order)</h3>
            <!-- AI: Create a prioritized list of 5–7 key stakeholder roles.
                 - Priority: Use numeric order (1 = highest). Assign High / Medium / Low tags.
                 - Role: Formal role title or common variant (e.g., "Head of L&D / CLO").
                 - Function: Short descriptor of stakeholder type (Champion, Decision Maker, Gatekeeper, Compliance, Operations).
                 - Key Responsibility: Clear description of what this stakeholder controls in the procurement or pilot process. -->
            <table class="stakeholder-table">
                <thead>
                    <tr>
                        <th>Priority</th>
                        <th>Role</th>
                        <th>Function</th>
                        <th>Key Responsibility</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="tag tag-[[PRIORITY_LEVEL_1]]">1</span></td>
                        <td>[[STAKEHOLDER_ROLE_1]]</td>
                        <td>[[STAKEHOLDER_FUNCTION_1]]</td>
                        <td>[[STAKEHOLDER_RESPONSIBILITY_1]]</td>
                    </tr>
                    <tr>
                        <td><span class="tag tag-[[PRIORITY_LEVEL_2]]">2</span></td>
                        <td>[[STAKEHOLDER_ROLE_2]]</td>
                        <td>[[STAKEHOLDER_FUNCTION_2]]</td>
                        <td>[[STAKEHOLDER_RESPONSIBILITY_2]]</td>
                    </tr>
                    <tr>
                        <td><span class="tag tag-[[PRIORITY_LEVEL_3]]">3</span></td>
                        <td>[[STAKEHOLDER_ROLE_3]]</td>
                        <td>[[STAKEHOLDER_FUNCTION_3]]</td>
                        <td>[[STAKEHOLDER_RESPONSIBILITY_3]]</td>
                    </tr>
                    <tr>
                        <td><span class="tag tag-[[PRIORITY_LEVEL_4]]">4</span></td>
                        <td>[[STAKEHOLDER_ROLE_4]]</td>
                        <td>[[STAKEHOLDER_FUNCTION_4]]</td>
                        <td>[[STAKEHOLDER_RESPONSIBILITY_4]]</td>
                    </tr>
                    <tr>
                        <td><span class="tag tag-[[PRIORITY_LEVEL_5]]">5</span></td>
                        <td>[[STAKEHOLDER_ROLE_5]]</td>
                        <td>[[STAKEHOLDER_FUNCTION_5]]</td>
                        <td>[[STAKEHOLDER_RESPONSIBILITY_5]]</td>
                    </tr>
                    <tr>
                        <td><span class="tag tag-[[PRIORITY_LEVEL_6]]">6</span></td>
                        <td>[[STAKEHOLDER_ROLE_6]]</td>
                        <td>[[STAKEHOLDER_FUNCTION_6]]</td>
                        <td>[[STAKEHOLDER_RESPONSIBILITY_6]]</td>
                    </tr>
                </tbody>
            </table>

            <div class="risk-warning">
                <h4>Critical Gap: Contact Discovery Required</h4>
                <!-- AI: Clearly state what information is missing (e.g., names, emails, LinkedIn URLs).
                     - Provide next-step actions to resolve (e.g., LinkedIn Sales Navigator, referrals, vendor introductions).
                     - Keep tone factual and action-oriented. -->
                <p>[[CONTACT_DISCOVERY_GAP]]</p>
            </div>
        </section>


<!-- Risk Register -->
<section id="risk-register">
    <h2>Risk Register & Mitigations</h2>

    <!-- 
        INSTRUCTIONS FOR AI AGENT:
        - Populate this section with 3–7 risk items depending on the complexity of the case.
        - Each risk-item should include:
            (a) Title of the risk [[RISK_TITLE_n]] (short, descriptive, business-language).
            (b) Likelihood level [[RISK_LIKELIHOOD_n]] (High / Medium / Low).
            (c) Impact level [[RISK_IMPACT_n]] (High / Medium / Low).
            (d) Mitigation strategy [[RISK_MITIGATION_n]] (clear, actionable, concise).
        - Ensure coverage of:
            • At least one strategic risk (e.g., sponsor/market misalignment).
            • At least one regulatory/legal risk (if applicable).
            • At least one operational/logistics risk.
        - Write mitigations in proactive, action-oriented language.
        - Keep formatting consistent for readability.
    -->

    <div class="risk-matrix">

        <div class="risk-item risk-[[RISK_SEVERITY_1]]">
            <h4>[[RISK_TITLE_1]]</h4>
            <p><strong>Likelihood:</strong> [[RISK_LIKELIHOOD_1]] | 
               <strong>Impact:</strong> [[RISK_IMPACT_1]]</p>
            <p><strong>Mitigation:</strong> [[RISK_MITIGATION_1]]</p>
        </div>

        <div class="risk-item risk-[[RISK_SEVERITY_2]]">
            <h4>[[RISK_TITLE_2]]</h4>
            <p><strong>Likelihood:</strong> [[RISK_LIKELIHOOD_2]] | 
               <strong>Impact:</strong> [[RISK_IMPACT_2]]</p>
            <p><strong>Mitigation:</strong> [[RISK_MITIGATION_2]]</p>
        </div>

        <div class="risk-item risk-[[RISK_SEVERITY_3]]">
            <h4>[[RISK_TITLE_3]]</h4>
            <p><strong>Likelihood:</strong> [[RISK_LIKELIHOOD_3]] | 
               <strong>Impact:</strong> [[RISK_IMPACT_3]]</p>
            <p><strong>Mitigation:</strong> [[RISK_MITIGATION_3]]</p>
        </div>

        <!-- Add more risk-item blocks as required -->
    </div>
</section>


        <!-- Outreach Plan & Logistics -->
        <section id="outreach-plan">
            <h2>Outreach Plan, Pilot Design & Logistics</h2>
            
            <!-- AI: Generate a 6-step outreach sequence.
                 - Each step should have: Step number (badge), Step title, and short actionable description.
                 - Keep sequence logical: starts with warm intro → discovery → pilot → reporting.
                 - Tailor tone for B2B enterprise outreach. -->
            <h3>6-Touch Outreach Sequence</h3>
            <div class="next-steps-timeline">
                <div class="timeline-item">
                    <div class="timeline-badge">1</div>
                    <div>
                        <strong>[[STEP_1_TITLE]]:</strong> [[STEP_1_DESCRIPTION]]
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-badge">2</div>
                    <div>
                        <strong>[[STEP_2_TITLE]]:</strong> [[STEP_2_DESCRIPTION]]
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-badge">3</div>
                    <div>
                        <strong>[[STEP_3_TITLE]]:</strong> [[STEP_3_DESCRIPTION]]
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-badge">4</div>
                    <div>
                        <strong>[[STEP_4_TITLE]]:</strong> [[STEP_4_DESCRIPTION]]
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-badge">5</div>
                    <div>
                        <strong>[[STEP_5_TITLE]]:</strong> [[STEP_5_DESCRIPTION]]
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-badge">6</div>
                    <div>
                        <strong>[[STEP_6_TITLE]]:</strong> [[STEP_6_DESCRIPTION]]
                    </div>
                </div>
            </div>

            <!-- AI: Provide 1 sample cold outreach message in LinkedIn/Email style.
                 - Keep it short, credible, and pilot-focused.
                 - Should highlight ROI, pilot duration, and low-risk commitment. -->
            <div class="section-highlight">
                <h4>Sample LinkedIn/Email Opener</h4>
                <p><em>"[[OUTREACH_OPENER]]"</em></p>
            </div>

            <!-- AI: Fill in pricing/commercial offer grid.
                 - Pilot sizes: Small, Medium, Large if available.
                 - Learners: Numeric ranges or set counts.
                 - Cost estimate: Currency and range.
                 - Lead time: Shipping/ops estimate in days. -->
            <h3>Pilot Pricing & Commercial Offer</h3>
            <div class="data-grid">
                <div class="data-card">
                    <div class="metric-value">[[PILOT_SIZE_1]]</div>
                    <div class="metric-label">[[LEARNERS_1]]</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[PILOT_SIZE_2]]</div>
                    <div class="metric-label">[[LEARNERS_2]]</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[COST_RANGE]]</div>
                    <div class="metric-label">Per-kit Cost Estimate</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[LEAD_TIME]]</div>
                    <div class="metric-label">Lead Time Estimate</div>
                </div>
            </div>

            <!-- AI: Logistics guidance.
                 - List 4–6 local 3PL partners relevant to region (default Bangalore).
                 - Recommend 1 fulfillment model: Centralized, decentralized, hybrid.
                 - Include 1 tag: Reduces Complexity / Cost Efficient / Scalable. -->
            <h3>Logistics & Fulfillment (Bangalore)</h3>
            <div class="competitive-grid">
                <div class="competitor-card">
                    <h4>3PL Partners (Shortlist)</h4>
                    <ul class="bullet-points">
                        <li>[[3PL_PARTNER_1]]</li>
                        <li>[[3PL_PARTNER_2]]</li>
                        <li>[[3PL_PARTNER_3]]</li>
                        <li>[[3PL_PARTNER_4]]</li>
                        <li>[[3PL_PARTNER_5]]</li>
                        <li>[[3PL_PARTNER_6]]</li>
                    </ul>
                </div>
                <div class="competitor-card">
                    <h4>Recommended Fulfillment Model</h4>
                    <p>[[FULFILLMENT_MODEL_DESCRIPTION]]</p>
                    <div class="tag tag-low">[[FULFILLMENT_TAG]]</div>
                </div>
            </div>

            <!-- AI: Fill measurement plan in table.
                 - Phases: Pre-Assessment, Intervention, Post-Assessment, 30-day, 90-day.
                 - Each row = Phase | Measurement | Timeline | Deliverable.
                 - Ensure pilot impact is measurable in short & long term. -->
            <h3>Pilot Measurement Plan</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Phase</th>
                        <th>Measurement</th>
                        <th>Timeline</th>
                        <th>Deliverable</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>[[PHASE_1]]</td>
                        <td>[[MEASUREMENT_1]]</td>
                        <td>[[TIMELINE_1]]</td>
                        <td>[[DELIVERABLE_1]]</td>
                    </tr>
                    <tr>
                        <td>[[PHASE_2]]</td>
                        <td>[[MEASUREMENT_2]]</td>
                        <td>[[TIMELINE_2]]</td>
                        <td>[[DELIVERABLE_2]]</td>
                    </tr>
                    <tr>
                        <td>[[PHASE_3]]</td>
                        <td>[[MEASUREMENT_3]]</td>
                        <td>[[TIMELINE_3]]</td>
                        <td>[[DELIVERABLE_3]]</td>
                    </tr>
                    <tr>
                        <td>[[PHASE_4]]</td>
                        <td>[[MEASUREMENT_4]]</td>
                        <td>[[TIMELINE_4]]</td>
                        <td>[[DELIVERABLE_4]]</td>
                    </tr>
                    <tr>
                        <td>[[PHASE_5]]</td>
                        <td>[[MEASUREMENT_5]]</td>
                        <td>[[TIMELINE_5]]</td>
                        <td>[[DELIVERABLE_5]]</td>
                    </tr>
                </tbody>
            </table>
        </section>

<!-- Next Steps -->
<section id="next-steps">
    <h2>Next Steps, Ownership & Implementation</h2>
    
    <!-- ================= Immediate Actions (7–14 Days) ================= -->
    <h3>Immediate Actions (Next 7–14 Days)</h3>
    <div class="key-findings">
        
        <!-- Finding Card Template -->
        <div class="finding-card [[PRIORITY_CLASS]]">
            <!-- [[PRIORITY_CLASS]] can be "critical", "opportunity", or leave empty -->
            <h4>[[ACTION_AREA]]</h4> 
            <!-- Example: "Legal & Compliance", "Sales Materials", "Contact Discovery" -->
            
            <p><strong>Owner:</strong> [[OWNER_ROLE]]</p>
            <!-- Example: "Legal/Compliance + Sales Intelligence", "Sales SDR", etc. -->
            
            <ul class="bullet-points">
                <!-- Fill in 2–4 immediate tactical actions -->
                <li>[[IMMEDIATE_ACTION_1]]</li>
                <li>[[IMMEDIATE_ACTION_2]]</li>
                <li>[[IMMEDIATE_ACTION_3]]</li>
            </ul>
        </div>
        <!-- Repeat multiple finding-card blocks as needed -->
        
    </div>
    
    <!-- ================= Medium Term (1–3 Months) ================= -->
    <h3>Medium Term (1–3 Months)</h3>
    <div class="next-steps-timeline">
        
        <!-- Timeline Item Template -->
        <div class="timeline-item">
            <div class="timeline-badge">[[OWNER_ABBR]]</div>
            <!-- [[OWNER_ABBR]] = short role label, e.g., AE (Account Executive), CS (Customer Success), SE (Sales Engineer) -->
            
            <div>
                <strong>[[MEDIUM_TERM_ACTION_TITLE]]:</strong> [[MEDIUM_TERM_ACTION_DESC]]
                <!-- Example: "Outreach Execution: Run outreach cadence to target leads; secure pilot SOW" -->
            </div>
        </div>
        <!-- Repeat multiple timeline-item blocks as needed -->
        
    </div>
    
    <!-- ================= Long Term (3–12 Months) ================= -->
    <h3>Long Term (3–12 Months)</h3>
    <div class="recommendation-box">
        <ul class="bullet-points">
            <!-- Focus on 3–5 strategic growth objectives -->
            <li><strong>[[LONG_TERM_STRATEGY_1_TITLE]]:</strong> [[LONG_TERM_STRATEGY_1_DESC]]</li>
            <li><strong>[[LONG_TERM_STRATEGY_2_TITLE]]:</strong> [[LONG_TERM_STRATEGY_2_DESC]]</li>
            <li><strong>[[LONG_TERM_STRATEGY_3_TITLE]]:</strong> [[LONG_TERM_STRATEGY_3_DESC]]</li>
        </ul>
    </div>
</section>




<!-- Appendices -->
<!-- 
    INSTRUCTIONS FOR AI AGENT:
    - Replace [[COMPANY]] and [[CITY]] with the target organization details.
    - Keep all outputs structured, minimal, and export-friendly (usable in CSV, Excel, or Docs).
    - Each appendix serves as a ready-to-use artifact; do not add narrative text.
-->
<section id="appendices">
    <h2>Appendices & Practical Artifacts</h2>
    
    <div class="appendix-grid">
        
        <!-- Appendix A: Outreach Tracker -->
        <div class="appendix-card">
            <h4>A. Outreach Tracker Template</h4>
            <p><strong>CSV Columns:</strong></p>
            <ul class="bullet-points">
                <!-- List exactly the columns to appear in the CSV -->
                <li>[[OUTREACH_COLUMN_1]]</li>
                <li>[[OUTREACH_COLUMN_2]]</li>
                <li>[[OUTREACH_COLUMN_3]]</li>
                <li>[[OUTREACH_COLUMN_4]]</li>
                <li>[[OUTREACH_COLUMN_5]]</li>
                <li>[[OUTREACH_COLUMN_6]]</li>
                <li>[[OUTREACH_COLUMN_7]]</li>
            </ul>
            <!-- Guidance:
                 - Include fields like persona name, role, contact method, date, status, notes, next action, etc.
                 - Columns must stay consistent across projects for easy CSV export.
            -->
        </div>
        
        <!-- Appendix B: Pilot Evaluation -->
        <div class="appendix-card">
            <h4>B. Pilot Evaluation Template</h4>
            <p><strong>KPIs to Track:</strong></p>
            <ul class="bullet-points">
                <li>[[PILOT_KPI_1]]</li>
                <li>[[PILOT_KPI_2]]</li>
                <li>[[PILOT_KPI_3]]</li>
                <li>[[PILOT_KPI_4]]</li>
            </ul>
            <!-- Guidance:
                 - Focus on learning impact, adoption, and satisfaction.
                 - Include both short-term (delta scores, engagement) and long-term metrics (retention, NPS).
            -->
        </div>
        
        <!-- Appendix C: Compliance Checklist -->
        <div class="appendix-card">
            <h4>C. Compliance Checklist</h4>
            <p><strong>Pilot Must Include:</strong></p>
            <ul class="bullet-points">
                <li>[[COMPLIANCE_ITEM_1]]</li>
                <li>[[COMPLIANCE_ITEM_2]]</li>
                <li>[[COMPLIANCE_ITEM_3]]</li>
                <li>[[COMPLIANCE_ITEM_4]]</li>
                <li>[[COMPLIANCE_ITEM_5]]</li>
            </ul>
            <!-- Guidance:
                 - Adapt checklist to local regulations (e.g., DPDP, GDPR).
                 - Cover consent, purpose limitation, minimization, breach notifications, retention, rights handling.
            -->
        </div>
        
        <!-- Appendix D: Logistics Worksheet -->
        <div class="appendix-card">
            <h4>D. Logistics Cost Worksheet</h4>
            <p><strong>BOM Lines:</strong></p>
            <ul class="bullet-points">
                <li>[[LOGISTICS_ITEM_1]]</li>
                <li>[[LOGISTICS_ITEM_2]]</li>
                <li>[[LOGISTICS_ITEM_3]]</li>
                <li>[[LOGISTICS_ITEM_4]]</li>
            </ul>
            <!-- Guidance:
                 - Capture key cost drivers: materials, packaging, warehousing, delivery.
                 - Keep list concise and quantifiable for costing analysis.
            -->
        </div>
        
        <!-- Appendix E: LinkedIn Search -->
        <div class="appendix-card">
            <h4>E. LinkedIn Search Templates</h4>
            <p><strong>Query Strings:</strong></p>
            
            <code style="background: #f8f9fa; padding: 10px; border-radius: 5px; display: block; margin: 10px 0;">
                Title:"[[ROLE_1]]" Company:"[[COMPANY]]" Location:"[[CITY]]"
            </code>
            
            <code style="background: #f8f9fa; padding: 10px; border-radius: 5px; display: block; margin: 10px 0;">
                Title:"[[ROLE_2]]" Company:"[[COMPANY]]" Location:"[[CITY]]"
            </code>
            <!-- Guidance:
                 - Generate 2–3 Boolean search queries for key buyer/sponsor roles.
                 - Roles vary per case: CLO, Head of L&D, AI COE Lead, CHRO, etc.
            -->
        </div>
        
        <!-- Appendix F: Web Search -->
        <div class="appendix-card">
            <h4>F. Web Search Templates</h4>
            <p><strong>Discovery Queries:</strong></p>
            <ul class="bullet-points">
                <li>"[[COMPANY]] [[DISCOVERY_KEYWORD_1]] [[CITY]]"</li>
                <li>"[[COMPANY]] [[DISCOVERY_KEYWORD_2]] [[CITY]]"</li>
            </ul>
            <!-- Guidance:
                 - Provide 2–3 queries focused on technical discovery, procurement, or partnerships.
                 - Tailor keywords to integration (e.g., platform, vendor onboarding, supplier registration).
            -->
        </div>
        
    </div>
</section>


<!-- Final Call to Action -->
<section id="final-notes">
    <h2>Final Notes & Immediate Asks</h2>
    
    <!-- 
        INSTRUCTIONS FOR AI AGENT:
        - This section should summarize the most urgent next steps.
        - Group them into 3–5 themed cards (e.g., Budget, Sales Enablement, Internal Routing).
        - Each card should have 2–4 bullet points written as actionable tasks.
        - Use [[Company]] and [[City]] placeholders where relevant.
        - Keep tasks concrete, like "Produce one-pager" or "Assign SDR for outreach," not vague.
        - Frame all content as recommendations for the [[Team]] (e.g., "Global Knowledge Tech Sales Team").
    -->
        <h3>For the [[Team]]</h3>
        
        <div class="key-findings">
            <!-- Card 1 -->
            <div class="finding-card critical">
                <h4>1. [[PRIORITY_ACTION_TITLE_1]]</h4>
                <ul class="bullet-points">
                    <li>[[ACTION_ITEM_1A]]</li>
                    <li>[[ACTION_ITEM_1B]]</li>
                    <li>[[ACTION_ITEM_1C]]</li>
                </ul>
            </div>
            
            <!-- Card 2 -->
            <div class="finding-card opportunity">
                <h4>2. [[PRIORITY_ACTION_TITLE_2]]</h4>
                <ul class="bullet-points">
                    <li>[[ACTION_ITEM_2A]]</li>
                    <li>[[ACTION_ITEM_2B]]</li>
                </ul>
            </div>
            
            <!-- Card 3 -->
            <div class="finding-card">
                <h4>3. [[PRIORITY_ACTION_TITLE_3]]</h4>
                <ul class="bullet-points">
                    <li>[[ACTION_ITEM_3A]]</li>
                    <li>[[ACTION_ITEM_3B]]</li>
                </ul>
            </div>
            
            <!-- OPTIONAL: Card 4 if needed -->
            <div class="finding-card">
                <h4>4. [[PRIORITY_ACTION_TITLE_4]]</h4>
                <ul class="bullet-points">
                    <li>[[ACTION_ITEM_4A]]</li>
                    <li>[[ACTION_ITEM_4B]]</li>
                </ul>
            </div>
        </div>
</section>

<!-- References Section -->
<div class="citation-footer">
    <h3>References</h3>

    <!-- 
        INSTRUCTIONS FOR AI AGENT:
        - Populate this section with 5–15 citations depending on the length of the report.
        - Each citation should follow the format:
              <strong>[n]</strong> <a href="[[SOURCE_URL]]">[[SOURCE_NAME]]</a> - [[SHORT_DESCRIPTION]]
        - Use sequential numbering [1], [2], [3]…
        - [[SOURCE_NAME]] should be the domain name (e.g., meity.gov.in, mckinsey.com).
        - [[SHORT_DESCRIPTION]] should briefly describe what was drawn from the source (e.g., "DPDP guidance and legal notes").
        - Prioritize authoritative references: government portals, regulatory bodies, research firms, company annual reports, reputable news outlets.
        - Ensure URLs are direct and accessible (not redirect links).
        - Include at least 1–2 competitor/company sources, 1 government/regulatory source, and 1 logistics/operations source where relevant.
    -->

    <div class="citation-item">
        <strong>[1]</strong> <a href="[[SOURCE_URL_1]]">[[SOURCE_NAME_1]]</a> - [[SHORT_DESCRIPTION_1]]
    </div>

    <div class="citation-item">
        <strong>[2]</strong> <a href="[[SOURCE_URL_2]]">[[SOURCE_NAME_2]]</a> - [[SHORT_DESCRIPTION_2]]
    </div>

    <div class="citation-item">
        <strong>[3]</strong> <a href="[[SOURCE_URL_3]]">[[SOURCE_NAME_3]]</a> - [[SHORT_DESCRIPTION_3]]
    </div>

    <!-- More citation-items as needed -->

    <p style="font-style: italic; margin-top: 15px;">
        Note: [[INCLUDE_LIMITATIONS_NOTE]]  
        <!-- Example: "Further competitive benchmarking required for pricing data." -->
    </p>
</div>

</body>
</html>

---
    ### HTML TEMPLATE COMPLIANCE

    **CRITICAL:** You must use the EXACT HTML structure provided above. Do not modify the HTML structure, CSS, or JavaScript. Only replace the bracketed placeholders with actual data.

    ** Missing Data Handling:**
    - If specific data is not found, explicitly state "Information not available in research"
    - Do not leave bracketed placeholders unfilled
    - For missing metrics, use "N/A" or state "Data not found"
    - For missing sections, include a note explaining the limitation

    ### WIKIPEDIA-STYLE CITATION REQUIREMENTS
    **Citation Format:** All numbered citations should be hyperlinks to the relevant reference at the bottom.
    - Cite all market size figures, growth rates, and financial data
    - Cite customer demographic and behavioral statistics  
    - Cite competitor information, market share data, and positioning claims
    - Cite industry trends, regulatory factors, and PESTLE analysis points
    - Citations will be automatically converted to numbered hyperlinks

    ---
    ### FINAL QUALITY CHECKLIST
    - [ ] All bracketed placeholders replaced with actual data
    - [ ] Tables populated with ALL real segment data (remove placeholder rows)
    - [ ] Metrics and KPIs updated with actual values
    - [ ] CSS variables set for visual elements (--v: 0-100 for bars, --x/--y: 0-100 for positioning)
    - [ ] Citations added for all factual claims
    - [ ] Missing data explicitly acknowledged where applicable
    - [ ] Professional tone maintained throughout
    - [ ] Strategic insights provided in each section

    MANDATES (must follow exactly)
    1. Preserve the entire HTML/CSS/JS structure exactly as provided. Do not add, remove, or re-order tags, style blocks, script tags, id attributes, or citation anchors. (If you modify structure, the output is invalid.)
    2. Replace **every** `[[BRACKETED_PLACEHOLDER]]` with content of the correct type:
    - If placeholder is a visible text block → replace with a concise human-ready sentence/paragraph.
    - If placeholder is a metric card → replace with value + units (e.g., "₹10,372 Cr | #ref15").
    - If placeholder is a JSON hook inside `<script type="application/json" id="...">` → **replace with valid JSON** only (no comments, no trailing commas). If unknown, insert a valid JSON object indicating missing data (see Missing Data rule).
    3. JSON rules:
    - Must be valid JSON (strict). Example: `[{"region":"Bengaluru","score":90,"label":"High (9/10)","citation":"#ref3"}]`.
    - If a value is estimated add `"estimate": true` in the object.
    - Percent strings must end with '%'. Currency must include symbol & magnitude (K, M, B, Cr).
    4. Missing data: **do not leave placeholders**. Use the exact text `"Information not available in research"` for human text fields. For JSON hooks use valid JSON with a single object: `{"note":"Information not available in research","estimate":true}`.
    5. Citation anchors must be preserved exactly (e.g., `<a href="#ref11" class="citation">[11]</a>`). When you state a fact, attach the appropriate citation anchor inline as in the template.
    6. Output format: **the response must be the complete HTML only** (one code block or raw HTML). No explanations, no extra JSON, no markdown headers. If the agent cannot fill a placeholder because the source data lacks it, still replace it (see rule 4).
    7. Final validation (automated checks your agent must pass before returning):
    - No `[[` or `]]` remain anywhere.
    - All `<script type="application/json" id="...">` blocks contain parseable JSON.
    - Percent values end with `%`.
    - Currency values include symbol and magnitude suffix (K/M/B/Cr).
    - Citation anchors exist where the template required them.
    Generate the complete HTML report using the template above with all placeholders filled with actual research data.


"""