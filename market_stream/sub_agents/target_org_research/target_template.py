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
    * Citation Sources: {sources}
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
    - Inline citation tags in the Markdown (e.g., `[12]`, `(ref: 12)`) should be preserved verbatim in the relevant sentences. The "References"/"Citations" section should render items `[n]` with `SOURCE_NAME` and URL if provided in Markdown. Do **not** fabricate URLs.
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
    <title>Organization Intelligence Report - [[ORG_NAME]]</title>
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
        
        .content-section {
            margin: 25px 0;
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
        }

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
            <h1>Organization Intelligence Report: [[ORG_NAME]]</h1>
            <div class="report-subtitle">Comprehensive Analysis of [[ORG_NAME]] Operations and Structure</div>
        </div>

        <div class="report-meta">
            <div><strong>Project ID:</strong> [[PROJECT_ID]]</div>
            <div><strong>Target Organization:</strong> [[ORG_NAME]]</div>
            <div><strong>Prepared by:</strong> [[AUTHOR]]</div>
            <div><strong>Date:</strong> [[REPORT_DATE]]</div>
        </div>

        <div class="section-highlight">
            <p><strong>Note on citations:</strong> [[CITATION_NOTE]]</p>
        </div>

        <!-- EXECUTIVE SUMMARY -->
        <section id="executive-summary" class="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="data-grid">
                <div class="data-card">
                    <div class="metric-value">[[ORG_SIZE]]</div>
                    <div class="metric-label">Organization Size</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[REVENUE]]</div>
                    <div class="metric-label">Annual Revenue</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[INDUSTRY]]</div>
                    <div class="metric-label">Primary Industry</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[HEADQUARTERS]]</div>
                    <div class="metric-label">Headquarters Location</div>
                </div>
            </div>

            <div class="content-section">
                <h4>Key Organizational Facts</h4>
                <ul class="bullet-points">
                    <li><strong>Founded:</strong> [[FOUNDED_YEAR]]</li>
                    <li><strong>Primary Markets:</strong> [[PRIMARY_MARKETS]]</li>
                    <li><strong>Key Products/Services:</strong> [[KEY_PRODUCTS]]</li>
                    <li><strong>Recent Developments:</strong> [[RECENT_DEVELOPMENTS]]</li>
                </ul>
            </div>
        </section>

        <!-- Organization Intelligence -->
        <section id="organization-intelligence">
            <h2>Organization Intelligence: [[ORG_NAME]]</h2>
            
            <div class="segment-grid">
                <div class="segment-card">
                    <h4>Company Fundamentals</h4>
                    <p>[[ORG_DESCRIPTION]]</p>

                    <div class="data-grid">
                        <div class="data-card">
                            <div class="metric-value">[[EMPLOYEE_COUNT]]</div>
                            <div class="metric-label">Total Employees</div>
                        </div>
                        <div class="data-card">
                            <div class="metric-value">[[MARKET_CAP]]</div>
                            <div class="metric-label">Market Capitalization</div>
                        </div>
                    </div>
                </div>
                
                <div class="segment-card">
                    <h4>Technology Stack</h4>
                    <ul class="bullet-points">
                        <li><strong>[[TECH_STACK_TYPE_1]]:</strong> [[TECH_STACK_DETAIL_1]]</li>
                        <li><strong>[[TECH_STACK_TYPE_2]]:</strong> [[TECH_STACK_DETAIL_2]]</li>
                        <li><strong>[[TECH_STACK_TYPE_3]]:</strong> [[TECH_STACK_DETAIL_3]]</li>
                    </ul>
                </div>
                
                <div class="segment-card">
                    <h4>Strategic Themes</h4>
                    <ul class="bullet-points">
                        <li>[[STRATEGIC_THEME_1]]</li>
                        <li>[[STRATEGIC_THEME_2]]</li>
                        <li>[[STRATEGIC_THEME_3]]</li>
                    </ul>
                </div>
            </div>

            <div class="content-section">
                <h4>Organizational Structure</h4>
                <ul class="bullet-points">
                    <li><strong>Reporting Structure:</strong> [[REPORTING_STRUCTURE]]</li>
                    <li><strong>Key Departments:</strong> [[KEY_DEPARTMENTS]]</li>
                    <li><strong>Geographic Presence:</strong> [[GEOGRAPHIC_PRESENCE]]</li>
                </ul>
            </div>
        </section>

        <!-- Leadership & Key Personnel -->
        <section id="leadership">
            <h2>Leadership & Key Personnel</h2>
            
            <table class="stakeholder-table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Title</th>
                        <th>Tenure</th>
                        <th>Background</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>[[EXECUTIVE_NAME_1]]</td>
                        <td>[[EXECUTIVE_TITLE_1]]</td>
                        <td>[[EXECUTIVE_TENURE_1]]</td>
                        <td>[[EXECUTIVE_BACKGROUND_1]]</td>
                    </tr>
                    <tr>
                        <td>[[EXECUTIVE_NAME_2]]</td>
                        <td>[[EXECUTIVE_TITLE_2]]</td>
                        <td>[[EXECUTIVE_TENURE_2]]</td>
                        <td>[[EXECUTIVE_BACKGROUND_2]]</td>
                    </tr>
                    <tr>
                        <td>[[EXECUTIVE_NAME_3]]</td>
                        <td>[[EXECUTIVE_TITLE_3]]</td>
                        <td>[[EXECUTIVE_TENURE_3]]</td>
                        <td>[[EXECUTIVE_BACKGROUND_3]]</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <!-- Financial Intelligence -->
        <section id="financial-intelligence">
            <h2>Financial Intelligence</h2>
            
            <div class="data-grid">
                <div class="data-card">
                    <div class="metric-value">[[REVENUE_TREND]]</div>
                    <div class="metric-label">Revenue Trend</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[PROFIT_MARGIN]]</div>
                    <div class="metric-label">Profit Margin</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[GROWTH_RATE]]</div>
                    <div class="metric-label">Growth Rate</div>
                </div>
            </div>

            <div class="content-section">
                <h4>Financial Highlights</h4>
                <ul class="bullet-points">
                    <li><strong>Recent Financial Performance:</strong> [[FINANCIAL_PERFORMANCE]]</li>
                    <li><strong>Key Financial Metrics:</strong> [[FINANCIAL_METRICS]]</li>
                    <li><strong>Investment Activities:</strong> [[INVESTMENT_ACTIVITIES]]</li>
                </ul>
            </div>
        </section>

        <!-- Market Position -->
        <section id="market-position">
            <h2>Market Position & Competitive Landscape</h2>
            
            <div class="competitive-grid">
                <div class="competitor-card">
                    <h4>Market Share</h4>
                    <p>[[MARKET_SHARE]]</p>
                </div>
                <div class="competitor-card">
                    <h4>Primary Competitors</h4>
                    <p>[[PRIMARY_COMPETITORS]]</p>
                </div>
                <div class="competitor-card">
                    <h4>Competitive Advantages</h4>
                    <p>[[COMPETITIVE_ADVANTAGES]]</p>
                </div>
            </div>

            <div class="content-section">
                <h4>Market Analysis</h4>
                <ul class="bullet-points">
                    <li><strong>Target Markets:</strong> [[TARGET_MARKETS]]</li>
                    <li><strong>Market Trends:</strong> [[MARKET_TRENDS]]</li>
                    <li><strong>Regulatory Environment:</strong> [[REGULATORY_ENVIRONMENT]]</li>
                </ul>
            </div>
        </section>

        <!-- Operations & Infrastructure -->
        <section id="operations">
            <h2>Operations & Infrastructure</h2>
            
            <div class="segment-grid">
                <div class="segment-card">
                    <h4>Facilities & Locations</h4>
                    <ul class="bullet-points">
                        <li>[[FACILITY_1]]</li>
                        <li>[[FACILITY_2]]</li>
                        <li>[[FACILITY_3]]</li>
                    </ul>
                </div>
                
                <div class="segment-card">
                    <h4>Operational Capabilities</h4>
                    <ul class="bullet-points">
                        <li>[[CAPABILITY_1]]</li>
                        <li>[[CAPABILITY_2]]</li>
                        <li>[[CAPABILITY_3]]</li>
                    </ul>
                </div>
                
                <div class="segment-card">
                    <h4>Supply Chain</h4>
                    <ul class="bullet-points">
                        <li>[[SUPPLIER_1]]</li>
                        <li>[[SUPPLIER_2]]</li>
                        <li>[[SUPPLIER_3]]</li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- References Section -->
        <div class="citation-footer">
            <h3>References</h3>

            <div class="citation-item">
                <strong>[1]</strong> <a href="[[SOURCE_URL_1]]">[[SOURCE_NAME_1]]</a> - [[SHORT_DESCRIPTION_1]]
            </div>

            <div class="citation-item">
                <strong>[2]</strong> <a href="[[SOURCE_URL_2]]">[[SOURCE_NAME_2]]</a> - [[SHORT_DESCRIPTION_2]]
            </div>

            <div class="citation-item">
                <strong>[3]</strong> <a href="[[SOURCE_URL_3]]">[[SOURCE_NAME_3]]</a> - [[SHORT_DESCRIPTION_3]]
            </div>

            <p style="font-style: italic; margin-top: 15px;">
                Note: [[INCLUDE_LIMITATIONS_NOTE]]  
            </p>
        </div>

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
    Generate the complete HTML report using the template above with all placeholders filled with actual research data.


"""