CON_TEMPLATE = """
    You are an expert market-context HTML report generator. You were given a fixed HTML template (do not alter it) that contains bracketed placeholders like [[PRODUCT_NAME]], [[REGIONAL_TABLE_JSON]], etc. Your job: **output only one artifact — the complete HTML file** with every bracketed placeholder replaced according to the rules below. Do not output commentary, analysis, or any extra text.

    

    INPUT
    - `market_research_findings` (text / bullet list) — use facts from here to populate placeholders.
    - `citations` (list of objects: {id, title, url, accessed}) — map these to reference anchors in the References section.
    - `market_intelligence_agent` (report-structure instructions) — follow if relevant.

    RETURN
    - Exactly one file: the completed HTML document string. No other output allowed.
    ---
    ### INPUT DATA SOURCES
    * Research Findings: {market_research_findings}
    * Citation Sources: {citations}
    * Report Structure: {market_intelligence_agent}

    ---
    ### HTML TEMPLATE
    Use this EXACT template structure, replace the bracketed placeholders and create the tables and charts with actual data:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Research Report</title>
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
            padding: 30px;
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
            font-size: 2.2em;
            margin: 0;
            font-weight: 700;
        }
        
        .report-subtitle {
            font-size: 1.1em;
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
            font-size: 1.6em;
        }
        
        h3 {
            color: #34495e;
            margin-top: 30px;
            font-size: 1.3em;
            border-left: 4px solid var(--secondary-color);
            padding-left: 15px;
        }
        
        .data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .data-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .data-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--secondary-color);
        }
        
        .data-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--box-shadow);
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--primary-color);
            margin: 15px 0;
            display: block;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .market-chart {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: var(--border-radius);
            padding: 20px;
            margin: 25px 0;
        }
        
        .chart-bar {
            display: flex;
            align-items: center;
            margin: 15px 0;
            position: relative;
        }
        
        .chart-label {
            min-width: 120px;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        .chart-track {
            flex: 1;
            height: 25px;
            background: #e9ecef;
            border-radius: 12px;
            margin: 0 15px;
            overflow: hidden;
            position: relative;
        }
        
        .chart-fill {
            height: 100%;
            border-radius: 12px;
            position: relative;
        }
        
        .tam-fill { background: #27ae60; }
        .sam-fill { background: #3498db; }
        .som-fill { background: #e74c3c; }
        
        .chart-value {
            min-width: 100px;
            text-align: right;
            font-weight: 600;
            color: var(--text-color);
        }
        
        .section-highlight {
            background: #e8f6ff;
            padding: 20px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--secondary-color);
            margin: 25px 0;
        }
        
        .risk-warning {
            background: #fff5f5;
            padding: 20px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--danger-color);
            margin: 25px 0;
        }
        
        .key-insights {
            background: #f0fff4;
            padding: 20px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--success-color);
            margin: 25px 0;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            border-radius: var(--border-radius);
            overflow: hidden;
        }
        
        .data-table th, .data-table td {
            border: 1px solid #dee2e6;
            padding: 12px 10px;
            text-align: left;
        }
        
        .data-table th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .data-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .data-table tr:hover {
            background-color: #e8f6ff;
        }
        
        .competitive-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        
        .competitor-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: var(--border-radius);
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .competitor-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .competitor-name {
            font-weight: bold;
            color: var(--primary-color);
            font-size: 1em;
            margin-bottom: 8px;
        }
        
        .competitor-focus {
            color: #7f8c8d;
            font-size: 0.85em;
            font-style: italic;
        }
        
        .risk-matrix {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 25px 0;
        }
        
        .risk-item {
            padding: 12px;
            border-radius: var(--border-radius);
            text-align: center;
            font-weight: 600;
            color: white;
        }
        
        .risk-high { background-color: var(--danger-color); }
        .risk-medium { background-color: var(--warning-color); }
        .risk-low { background-color: var(--success-color); }
        
        .citation {
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .citation:hover {
            text-decoration: underline;
        }
        
        .references {
            background: #f8f9fa;
            padding: 25px;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
            margin: 40px 0;
        }
        
        .references h2 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 10px;
        }
        
        .references ol {
            padding-left: 20px;
        }
        
        .references li {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .references li:last-child {
            border-bottom: none;
        }
        
        .references a {
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: 500;
        }
        
        .references a:hover {
            text-decoration: underline;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 0.85em;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            .report-meta {
                grid-template-columns: 1fr;
            }
            
            .data-grid {
                grid-template-columns: 1fr;
            }
            
            .competitive-grid {
                grid-template-columns: 1fr;
            }
            
            .risk-matrix {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="report-header">
            <h1>[[PRODUCT_NAME]]</h1>
            <p class="report-subtitle">[[REPORT_SUBTITLE]]</p>
            <div class="report-meta">
                <div><strong>Date:</strong> [[REPORT_DATE]]</div>
                <div><strong>Market Focus:</strong> [[MARKET_FOCUS]]</div>
                <div><strong>Industry:</strong> [[INDUSTRY_FOCUS]]</div>
                <div><strong>Analysis Period:</strong> [[ANALYSIS_PERIOD]]</div>
            </div>
        </div>

        <!-- EXECUTIVE SUMMARY -->
        <section id="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="section-highlight">
                <h3>Objective</h3>
                <p>[[REPORT_OBJECTIVE]]</p>
            </div>

            <div class="key-insights">
                <h3>Key Market Metrics</h3>
                
                <div class="data-grid">
                    <div class="data-card">
                        <span class="metric-value">[[TAM_VALUE]]</span>
                        <div class="metric-label">Total Addressable Market <a href="#ref1" class="citation">[1]</a></div>
                    </div>
                    <div class="data-card">
                        <span class="metric-value">[[SAM_VALUE]]</span>
                        <div class="metric-label">Serviceable Available Market <a href="#ref2" class="citation">[2]</a></div>
                    </div>
                    <div class="data-card">
                        <span class="metric-value">[[SOM_VALUE]]</span>
                        <div class="metric-label">Serviceable Obtainable Market <a href="#ref3" class="citation">[3]</a></div>
                    </div>
                    <div class="data-card">
                        <span class="metric-value">[[GROWTH_RATE]]</span>
                        <div class="metric-label">Projected Growth Rate <a href="#ref4" class="citation">[4]</a></div>
                    </div>
                </div>

                <h4>Strategic Implications</h4>
                <p>[[STRATEGIC_IMPLICATIONS]]</p>
            </div>

            <div class="market-chart">
                <h4>Market Sizing Visualization</h4>
                <div class="chart-bar">
                    <div class="chart-label">TAM</div>
                    <div class="chart-track">
                        <div class="chart-fill tam-fill" style="width: [[TAM_PERCENTAGE]]%;"></div>
                    </div>
                    <div class="chart-value">[[TAM_VALUE]]</div>
                </div>
                <div class="chart-bar">
                    <div class="chart-label">SAM</div>
                    <div class="chart-track">
                        <div class="chart-fill sam-fill" style="width: [[SAM_PERCENTAGE]]%;"></div>
                    </div>
                    <div class="chart-value">[[SAM_VALUE]]</div>
                </div>
                <div class="chart-bar">
                    <div class="chart-label">SOM</div>
                    <div class="chart-track">
                        <div class="chart-fill som-fill" style="width: [[SOM_PERCENTAGE]]%;"></div>
                    </div>
                    <div class="chart-value">[[SOM_VALUE]]</div>
                </div>
            </div>
        </section>

        <!-- MARKET OVERVIEW -->
        <section id="market-overview">
            <h2>Market Overview</h2>
            
            <div class="section-highlight">
                <p>[[MARKET_OVERVIEW_SUMMARY]] <a href="#ref1" class="citation">[1]</a></p>
            </div>

            <h3>Industry Classification</h3>
            <p>[[INDUSTRY_CLASSIFICATION]]</p>

            <h3>Market Drivers</h3>
            <div class="key-insights">
                <ul>
                    <li>[[MARKET_DRIVER_1]]</li>
                    <li>[[MARKET_DRIVER_2]]</li>
                    <li>[[MARKET_DRIVER_3]]</li>
                    <li>[[MARKET_DRIVER_4]]</li>
                </ul>
            </div>
        </section>

        <!-- COMPETITIVE LANDSCAPE -->
        <section id="competitive-landscape">
            <h2>Competitive Landscape</h2>
            
            <h3>Key Players</h3>
            <div class="competitive-grid">
                <div class="competitor-card">
                    <div class="competitor-name">[[COMPETITOR_1]]</div>
                    <div class="competitor-focus">[[COMPETITOR_1_FOCUS]]</div>
                </div>
                <div class="competitor-card">
                    <div class="competitor-name">[[COMPETITOR_2]]</div>
                    <div class="competitor-focus">[[COMPETITOR_2_FOCUS]]</div>
                </div>
                <div class="competitor-card">
                    <div class="competitor-name">[[COMPETITOR_3]]</div>
                    <div class="competitor-focus">[[COMPETITOR_3_FOCUS]]</div>
                </div>
                <div class="competitor-card">
                    <div class="competitor-name">[[COMPETITOR_4]]</div>
                    <div class="competitor-focus">[[COMPETITOR_4_FOCUS]]</div>
                </div>
            </div>

            <h3>Competitive Positioning</h3>
            <div class="wide-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Competitor</th>
                            <th>Market Share</th>
                            <th>Strengths</th>
                            <th>Weaknesses</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>[[COMPETITOR_1]]</td>
                            <td>[[COMPETITOR_1_SHARE]]</td>
                            <td>[[COMPETITOR_1_STRENGTHS]]</td>
                            <td>[[COMPETITOR_1_WEAKNESSES]]</td>
                        </tr>
                        <tr>
                            <td>[[COMPETITOR_2]]</td>
                            <td>[[COMPETITOR_2_SHARE]]</td>
                            <td>[[COMPETITOR_2_STRENGTHS]]</td>
                            <td>[[COMPETITOR_2_WEAKNESSES]]</td>
                        </tr>
                        <tr>
                            <td>[[COMPETITOR_3]]</td>
                            <td>[[COMPETITOR_3_SHARE]]</td>
                            <td>[[COMPETITOR_3_STRENGTHS]]</td>
                            <td>[[COMPETITOR_3_WEAKNESSES]]</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- RISK ANALYSIS -->
        <section id="risk-analysis">
            <h2>Risk Analysis</h2>
            
            <div class="risk-matrix">
                <div class="risk-item risk-high">[[HIGH_RISK_1]]</div>
                <div class="risk-item risk-medium">[[MEDIUM_RISK_1]]</div>
                <div class="risk-item risk-low">[[LOW_RISK_1]]</div>
                <div class="risk-item risk-high">[[HIGH_RISK_2]]</div>
                <div class="risk-item risk-medium">[[MEDIUM_RISK_2]]</div>
            </div>

            <h3>Key Risk Factors</h3>
            <div class="risk-warning">
                <ul>
                    <li><strong>[[RISK_FACTOR_1]]:</strong> [[RISK_DESCRIPTION_1]]</li>
                    <li><strong>[[RISK_FACTOR_2]]:</strong> [[RISK_DESCRIPTION_2]]</li>
                    <li><strong>[[RISK_FACTOR_3]]:</strong> [[RISK_DESCRIPTION_3]]</li>
                </ul>
            </div>
        </section>

        <!-- OPPORTUNITIES -->
        <section id="opportunities">
            <h2>Market Opportunities</h2>
            
            <div class="key-insights">
                <h3>Growth Areas</h3>
                <ul>
                    <li>[[OPPORTUNITY_1]]</li>
                    <li>[[OPPORTUNITY_2]]</li>
                    <li>[[OPPORTUNITY_3]]</li>
                    <li>[[OPPORTUNITY_4]]</li>
                </ul>
            </div>

            <h3>Strategic Recommendations</h3>
            <div class="section-highlight">
                <ol>
                    <li>[[RECOMMENDATION_1]]</li>
                    <li>[[RECOMMENDATION_2]]</li>
                    <li>[[RECOMMENDATION_3]]</li>
                </ol>
            </div>
        </section>

        <!-- REFERENCES -->
        <section id="references" class="references">
            <h2>References</h2>
            <ol>
                <li id="ref1"><a href=[[REFERENCE_1 URL]]>[[REFERENCE_1 WEBSITE]]</a>: [[REFERENCE_1 SHORT DESCRIPTION]]</li>
                <li id="ref1"><a href=[[REFERENCE_2 URL]]>[[REFERENCE_2 WEBSITE]]</a>: [[REFERENCE_2 SHORT DESCRIPTION]]</li>
                <li id="ref1"><a href=[[REFERENCE_3 URL]]>[[REFERENCE_3 WEBSITE]]</a>: [[REFERENCE_3 SHORT DESCRIPTION]]</li>
                <li id="ref1"><a href=[[REFERENCE_4 URL]]>[[REFERENCE_4 WEBSITE]]</a>: [[REFERENCE_4 SHORT DESCRIPTION]]</li>
                <li id="ref1"><a href=[[REFERENCE_5 URL]]>[[REFERENCE_5 WEBSITE]]</a>: [[REFERENCE_5 SHORT DESCRIPTION]]</li>
                <!--Add all other citations in the same format-->
            </ol>
        </section>

        <footer>
            <p>© [[REPORT_YEAR]] [[PRODUCT_NAME]] Market Analysis Report. All data points referenced to cited sources.</p>
        </footer>
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