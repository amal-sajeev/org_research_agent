SEG_TEMPLATE = """
    You are an expert segmentation analysis report writer specializing in creating comprehensive HTML reports that exactly follow the provided template structure.

    **MISSION:** Transform segmentation research data into a polished, professional HTML Segmentation Analysis Report following the exact template format with Wikipedia-style numbered citations.

    ---
    ### INPUT DATA SOURCES
    * Citation Sources: {citations}
    * Report Structure: {segmentation_intelligence_agent}

    ---
    ### HTML TEMPLATE
    Use this EXACT template structure and only replace the bracketed placeholders with actual data:

    ```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 
    TEMPLATE INSTRUCTIONS FOR AI AGENT:
    
    1. Replace [[COMPANY_NAME]] with the actual company name
    2. Replace [[REPORT_TITLE]] with the specific analysis title
    3. Replace [[DATE]] with the current date
    4. Replace [[FOCUS_MARKET]] with the target geographic market
    5. Replace all [[VARIABLE_NAME]] placeholders throughout the document
    6. Update market size figures, growth rates, and projections with actual data
    7. Replace segment names and descriptions with actual identified segments
    8. Update PESTLE analysis with relevant factors for the specific industry
    9. Replace competitive analysis with actual competitors
    10. Update all citations to point to real sources
    
    CITATION SYSTEM:
    - Use numbered citations like [1], [2], etc. that hyperlink to references
    - Each citation should link to #ref-X where X is the reference number
    - All references should be listed at the bottom in numerical order
    - Include actual URLs for web sources when possible
    -->
    <title>[[COMPANY_NAME]] - [[REPORT_TITLE]]: Market Segmentation Analysis</title>
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
        
        /* AI Agent Instructions: Use different colors for different segment types
           - enterprise: #3498db (blue)
           - smb: #2ecc71 (green) 
           - finance: #f1c40f (yellow)
           - academic: #9b59b6 (purple)
           - healthcare: #e74c3c (red)
           - technology: #34495e (dark blue-gray)
        */
        .segment-card.enterprise {
            border-top-color: #3498db;
        }
        
        .segment-card.smb {
            border-top-color: #2ecc71;
        }
        
        .segment-card.finance {
            border-top-color: #f1c40f;
        }
        
        .segment-card.academic {
            border-top-color: #9b59b6;
        }
        
        .segment-card.healthcare {
            border-top-color: #e74c3c;
        }
        
        .segment-card.technology {
            border-top-color: #34495e;
        }
        
        .segment-title {
            font-size: 1.3em;
            font-weight: bold;
            color: var(--primary-color);
            margin-top: 0;
        }
        
        .segment-meta {
            display: flex;
            justify-content: space-between;
            margin: 15px 0;
        }
        
        .segment-priority {
            background: var(--light-color);
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
        }
        
        /* PESTLE Analysis styling */
        .pestle-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .pestle-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: var(--border-radius);
            text-align: center;
            border-left: 4px solid var(--secondary-color);
        }
        
        .pestle-card.positive {
            border-left-color: var(--success-color);
        }
        
        .pestle-card.neutral {
            border-left-color: var(--warning-color);
        }
        
        .pestle-card.negative {
            border-left-color: var(--danger-color);
        }
        
        /* SWOT Analysis styling */
        .swot-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .swot-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .swot-card h4 {
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .swot-card ul {
            padding-left: 20px;
        }
        
        .swot-card.strengths {
            border-top: 4px solid var(--success-color);
        }
        
        .swot-card.weaknesses {
            border-top: 4px solid var(--danger-color);
        }
        
        .swot-card.opportunities {
            border-top: 4px solid var(--secondary-color);
        }
        
        .swot-card.threats {
            border-top: 4px solid var(--warning-color);
        }
        
        /* Positioning and Marketing Mix styling */
        .positioning-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
            margin: 20px 0;
            border-left: 4px solid var(--secondary-color);
        }
        
        .positioning-statement {
            font-style: italic;
            border-left: 3px solid var(--secondary-color);
            padding-left: 15px;
            margin: 15px 0;
        }
        
        .marketing-mix-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .marketing-mix-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .marketing-mix-card h4 {
            color: var(--secondary-color);
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #e9ecef;
        }
        
        /* References styling - Wikipedia-style numbered list */
        .reference-list {
            list-style: none;
            padding: 0;
            margin: 20px 0;
        }
        
        .reference-list li {
            margin: 12px 0;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .reference-list li:target {
            background-color: #fffacd;
            padding: 8px;
            border-radius: 4px;
        }
        
        .reference-number {
            font-weight: bold;
            color: var(--primary-color);
            margin-right: 8px;
            display: inline-block;
            min-width: 30px;
        }
        
        .reference-url {
            color: var(--secondary-color);
            word-break: break-all;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            .toc-list {
                columns: 1;
            }
            
            .report-meta {
                grid-template-columns: 1fr;
            }
            
            .data-grid {
                grid-template-columns: 1fr;
            }
            
            .key-findings {
                grid-template-columns: 1fr;
            }
            
            .segment-grid {
                grid-template-columns: 1fr;
            }
            
            .pestle-grid {
                grid-template-columns: 1fr;
            }
            
            .swot-container {
                grid-template-columns: 1fr;
            }
            
            .marketing-mix-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 
        HEADER SECTION INSTRUCTIONS:
        Replace [[COMPANY_NAME]] with the actual company name
        Replace [[REPORT_TITLE]] with the specific analysis title
        Update the subtitle to reflect the actual scope of analysis
        -->
        <header class="report-header">
            <h1>[[COMPANY_NAME]] - [[REPORT_TITLE]]</h1>
            <div class="report-subtitle">Market Segmentation Analysis & Strategic Recommendations</div>
        </header>

        <!-- 
        REPORT METADATA INSTRUCTIONS:
        Update the date, focus market, and expansion plan with actual information
        Add additional metadata fields as needed (e.g., industry, analysis period, etc.)
        -->
        <div class="report-meta">
            <div>
                <strong>Date:</strong> [[DATE]]
            </div>
            <div>
                <strong>Focus Market:</strong> [[FOCUS_MARKET]]
            </div>
            <div>
                <strong>Expansion Plan:</strong> [[EXPANSION_PLAN]]
            </div>
        </div>

        <!-- 
        EXECUTIVE SUMMARY SECTION INSTRUCTIONS:
        
        1. Update the core objective to reflect the actual business goals
        2. Replace market size figures with actual data and proper citations
        3. Update the key segments with actual identified segments
        4. Modify the strategic recommendations based on analysis findings
        5. Ensure all statements have proper numbered citations linking to references
        
        CITATION FORMAT: Use <a href="#ref-1" class="citation-link">[1]</a> for each citation
        -->
        <section id="executive-summary">
            <h2>1. Executive Summary: The Strategic Landscape at a Glance</h2>
            
            <div class="key-insights">
                <h4>Core Objective</h4>
                <p>To decode the market landscape for [[COMPANY_NAME]] - [[REPORT_TITLE]] and identify the highest-potential customer segments for focused GTM investment, starting in [[FOCUS_MARKET]] with a plan to expand [[EXPANSION_PLAN]]. The analysis is designed to inform segment-specific product, pricing, and marketing decisions that drive ROI and learning outcomes.<a href="#ref-1" class="citation-link">[1]</a></p>
            </div>

            <h3>The Market in Brief</h3>
            <!-- 
            AI AGENT INSTRUCTIONS: 
            Replace the following paragraph with actual market context for your specific industry and geography.
            Include specific market drivers, key trends, and growth factors.
            Update all market size figures with real data and proper citations.
            -->
            <p>The [[INDUSTRY_NAME]] market in [[TARGET_GEOGRAPHY]] is expanding rapidly, underpinned by [[KEY_MARKET_DRIVERS]]. [[FOCUS_MARKET]] is positioned as a strategic starting point due to [[STRATEGIC_ADVANTAGES]]. Growth analyses show multi-year expansion with sizable [[REVENUE_METRIC]] potential depending on segment and deployment scope.<a href="#ref-2" class="citation-link">[2]</a><a href="#ref-3" class="citation-link">[3]</a><a href="#ref-4" class="citation-link">[4]</a></p>

            <!-- 
            DATA CARDS INSTRUCTIONS:
            Update these metrics with actual market data
            Ensure all figures are properly sourced with citations
            Consider adding additional metrics relevant to your specific market
            -->
            <div class="data-grid">
                <div class="data-card">
                    <div class="metric-value">[[MARKET_SIZE_CURRENT]]</div>
                    <div class="metric-label">[[CURRENT_YEAR]] Market Size</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[MARKET_SIZE_PROJECTION]]</div>
                    <div class="metric-label">[[PROJECTION_YEAR]] Projection</div>
                </div>
                <div class="data-card">
                    <div class="metric-value">[[CAGR_PERCENTAGE]]%</div>
                    <div class="metric-label">CAGR ([[CURRENT_YEAR]]-[[PROJECTION_YEAR]])</div>
                </div>
            </div>

            <!-- 
            MARKET ANALYSIS PARAGRAPH INSTRUCTIONS:
            Replace with actual market analysis including multiple data sources
            Provide range of estimates when available
            Include proper citations for all claims
            -->
            <p>Market size and growth signals include analyses projecting the [[MARKET_CATEGORY]] space in [[TARGET_GEOGRAPHY]] rising from [[CURRENT_MARKET_DESCRIPTION]] toward [[FUTURE_MARKET_DESCRIPTION]] by the end of the decade, with CAGR ranges commonly cited around [[CAGR_RANGE]] depending on scope (e.g., [[MARKET_SCOPE_EXAMPLES]]). For instance, select market estimates indicate [[SPECIFIC_ESTIMATE_1]] growing toward [[SPECIFIC_ESTIMATE_2]] (CAGR ~[[CAGR_1]]%), while other analyses project [[ALTERNATIVE_ESTIMATE]] (CAGR ~[[CAGR_2]]%).<a href="#ref-1" class="citation-link">[1]</a><a href="#ref-2" class="citation-link">[2]</a></p>

            <p>[[ADOPTION_TRENDS_DESCRIPTION]] are increasingly highlighted in [[TARGET_GEOGRAPHY]]'s [[INDUSTRY_TYPE]] landscape, with [[LEADING_SECTORS]] leading early adoption. [[TARGET_CUSTOMER_TYPE]] are actively investing in [[CAPABILITY_AREA]], signaling strong demand for [[SOLUTION_CATEGORY]].<a href="#ref-5" class="citation-link">[5]</a><a href="#ref-6" class="citation-link">[6]</a></p>

            <!-- 
            KEY SEGMENTS INSTRUCTIONS:
            Replace these segment cards with your actual identified segments
            Use appropriate CSS classes for segment types (enterprise, smb, finance, academic, healthcare, technology)
            Update titles and descriptions based on your analysis
            Consider including 3-6 segments depending on market complexity
            -->
            <h3>Key Segments Identified</h3>
            <div class="segment-grid">
                <div class="segment-card [[SEGMENT_1_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_1_NAME]]</h4>
                    <p>[[SEGMENT_1_DESCRIPTION]]</p>
                </div>
                <div class="segment-card [[SEGMENT_2_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_2_NAME]]</h4>
                    <p>[[SEGMENT_2_DESCRIPTION]]</p>
                </div>
                <div class="segment-card [[SEGMENT_3_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_3_NAME]]</h4>
                    <p>[[SEGMENT_3_DESCRIPTION]]</p>
                </div>
                <div class="segment-card [[SEGMENT_4_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_4_NAME]]</h4>
                    <p>[[SEGMENT_4_DESCRIPTION]]</p>
                </div>
            </div>

            <!-- 
            STRATEGIC RECOMMENDATIONS INSTRUCTIONS:
            Update recommendation based on your actual analysis
            Include specific rationale and supporting citations
            Focus on actionable insights
            -->
            <div class="recommendation-box">
                <h4>Primary Target Recommendation</h4>
                <p>Prioritize [[PRIMARY_TARGETING_STRATEGY]] with [[PRIMARY_SEGMENT]] as the primary target due to [[PRIMARY_RATIONALE]], while actively pursuing [[SECONDARY_SEGMENT]] as a secondary target to [[SECONDARY_RATIONALE]]. The [[ADDITIONAL_SEGMENTS]] provide [[STRATEGIC_VALUE]] for longer-term growth.<a href="#ref-7" class="citation-link">[7]</a></p>
            </div>

            <div class="section-highlight">
                <h4>Critical Strategic Insight</h4>
                <p>The strongest early advantage stems from [[COMPETITIVE_ADVANTAGE_DESCRIPTION]]. This creates a layered competitive moat around [[COMPANY_NAME]]'s differentiator.<a href="#ref-8" class="citation-link">[8]</a></p>
            </div>

            <p><strong>Executive outlook:</strong> The plan emphasizes [[GTM_STRATEGY_SUMMARY]].<a href="#ref-9" class="citation-link">[9]</a></p>

            <!-- 
            DELIVERABLES SECTION INSTRUCTIONS:
            Update this section to reflect the actual deliverables for your analysis
            Include specific deliverable descriptions and formats
            -->
            <h3>Deliverables Preview</h3>
            <ul class="bullet-points">
                <li><strong>Deliverable 1:</strong> [[DELIVERABLE_1_DESCRIPTION]]</li>
                <li><strong>Deliverable 2:</strong> [[DELIVERABLE_2_DESCRIPTION]]</li>
                <li><strong>Deliverable 3:</strong> [[DELIVERABLE_3_DESCRIPTION]]</li>
                <li><strong>Additional deliverables:</strong> [[ADDITIONAL_DELIVERABLES_DESCRIPTION]]</li>
            </ul>
        </section>

        <!-- 
        MARKET OVERVIEW SECTION INSTRUCTIONS:
        
        1. Conduct a PESTLE analysis specific to your industry and geography
        2. Update each PESTLE factor with relevant analysis
        3. Use appropriate CSS classes (positive, neutral, negative) for each factor
        4. Include specific implications for each factor
        5. Add proper citations for all claims
        -->
        <section id="market-overview">
            <h2>2. Market Overview & Macro-Environment (PESTLE Analysis)</h2>
            
            <div class="pestle-grid">
                <div class="pestle-card [[POLITICAL_IMPACT_CLASS]]">
                    <h4>Political</h4>
                    <p>[[POLITICAL_ANALYSIS]]</p>
                    <p><strong>Implication:</strong> [[POLITICAL_IMPLICATION]]<a href="#ref-10" class="citation-link">[10]</a></p>
                </div>
                <div class="pestle-card [[ECONOMIC_IMPACT_CLASS]]">
                    <h4>Economic</h4>
                    <p>[[ECONOMIC_ANALYSIS]]</p>
                    <p><strong>Implication:</strong> [[ECONOMIC_IMPLICATION]]<a href="#ref-11" class="citation-link">[11]</a></p>
                </div>
                <div class="pestle-card [[SOCIAL_IMPACT_CLASS]]">
                    <h4>Social</h4>
                    <p>[[SOCIAL_ANALYSIS]]<a href="#ref-12" class="citation-link">[12]</a></p>
                </div>
                <div class="pestle-card [[TECHNOLOGICAL_IMPACT_CLASS]]">
                    <h4>Technological</h4>
                    <p>[[TECHNOLOGICAL_ANALYSIS]]<a href="#ref-13" class="citation-link">[13]</a></p>
                </div>
                <div class="pestle-card [[LEGAL_IMPACT_CLASS]]">
                    <h4>Legal</h4>
                    <p>[[LEGAL_ANALYSIS]]<a href="#ref-14" class="citation-link">[14]</a></p>
                </div>
                <div class="pestle-card [[ENVIRONMENTAL_IMPACT_CLASS]]">
                    <h4>Environmental</h4>
                    <p>[[ENVIRONMENTAL_ANALYSIS]]<a href="#ref-15" class="citation-link">[15]</a></p>
                </div>
            </div>

            <!-- 
            PESTLE IMPACT TABLE INSTRUCTIONS:
            Update this table with your actual PESTLE analysis results
            Use appropriate impact classifications and key considerations
            -->
            <h3>PESTLE Impact Summary</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>Impact</th>
                        <th>Key Considerations</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Political</td>
                        <td>[[POLITICAL_IMPACT]]</td>
                        <td>[[POLITICAL_CONSIDERATIONS]]<a href="#ref-10" class="citation-link">[10]</a></td>
                    </tr>
                    <tr>
                        <td>Economic</td>
                        <td>[[ECONOMIC_IMPACT]]</td>
                        <td>[[ECONOMIC_CONSIDERATIONS]]<a href="#ref-11" class="citation-link">[11]</a></td>
                    </tr>
                    <tr>
                        <td>Social</td>
                        <td>[[SOCIAL_IMPACT]]</td>
                        <td>[[SOCIAL_CONSIDERATIONS]]<a href="#ref-12" class="citation-link">[12]</a></td>
                    </tr>
                    <tr>
                        <td>Technological</td>
                        <td>[[TECHNOLOGICAL_IMPACT]]</td>
                        <td>[[TECHNOLOGICAL_CONSIDERATIONS]]<a href="#ref-13" class="citation-link">[13]</a></td>
                    </tr>
                    <tr>
                        <td>Legal</td>
                        <td>[[LEGAL_IMPACT]]</td>
                        <td>[[LEGAL_CONSIDERATIONS]]<a href="#ref-14" class="citation-link">[14]</a></td>
                    </tr>
                    <tr>
                        <td>Environmental</td>
                        <td>[[ENVIRONMENTAL_IMPACT]]</td>
                        <td>[[ENVIRONMENTAL_CONSIDERATIONS]]<a href="#ref-15" class="citation-link">[15]</a></td>
                    </tr>
                </tbody>
            </table>

            <!-- 
            MARKET SIZE SECTION INSTRUCTIONS:
            Update with actual market growth indicators and trends
            Include specific sector adoption patterns
            -->
            <h3>Market Size & Growth Signals</h3>
            <p>[[MARKET_GROWTH_DESCRIPTION]] indicate robust momentum, with [[ADOPTION_PATTERNS]] expanding and [[SPECIFIC_TRENDS]] scaling in [[LEADING_SECTORS]].<a href="#ref-16" class="citation-link">[16]</a><a href="#ref-17" class="citation-link">[17]</a></p>
        </section>

        <!-- 
        COMPETITIVE LANDSCAPE SECTION INSTRUCTIONS:
        
        1. Identify actual direct and indirect competitors
        2. Analyze competitive positioning and pricing
        3. Include Porter's Five Forces analysis
        4. Add regulatory and market trend considerations
        5. Ensure all analysis is properly cited
        -->
        <section id="competitive-landscape">
            <h2>3. Competitive Landscape: The Arena of Play</h2>
            
            <h3>Direct & Indirect Competitors</h3>
            <p><strong>Direct:</strong> [[DIRECT_COMPETITORS]]; adjacent players include [[ADJACENT_COMPETITORS]].<a href="#ref-18" class="citation-link">[18]</a></p>
            <p><strong>Indirect:</strong> [[INDIRECT_COMPETITORS]] representing alternatives to [[PRIMARY_SOLUTION_CATEGORY]].<a href="#ref-19" class="citation-link">[19]</a></p>

            <h3>Competitive Positioning Signals</h3>
            <p>Market observations show [[KEY_COMPETITOR_POSITIONING]] and pricing ([[PRICING_EXAMPLES]]). This pricing anchors [[TARGET_CUSTOMER]] expectations in [[TARGET_GEOGRAPHY]] and aligns with [[BUDGET_CONTEXT]].<a href="#ref-20" class="citation-link">[20]</a></p>
            <p>[[GEOGRAPHIC_FOCUS]]-focused [[TARGET_SEGMENT]] potential is considered [[MARKET_POTENTIAL_ASSESSMENT]], with major platforms viewing [[TARGET_GEOGRAPHY]] as [[GROWTH_OPPORTUNITY_DESCRIPTION]].<a href="#ref-21" class="citation-link">[21]</a></p>

            <h3>Regulatory & Market Trends</h3>
            <p>[[REGULATORY_TRENDS]] influence [[CUSTOMER_PROCUREMENT]] and [[BUSINESS_CONSIDERATIONS]]; [[SPECIFIC_REGULATIONS]] are evolving and relevant to [[BUSINESS_OPERATIONS]].<a href="#ref-22" class="citation-link">[22]</a></p>

            <!-- 
            PORTER'S FIVE FORCES INSTRUCTIONS:
            Analyze each force specific to your industry and market
            Provide implications for competitive strategy
            -->
            <h3>Porter's Five Forces (Strategic Implications)</h3>
            <ul class="bullet-points">
                <li><strong>Threat of new entrants:</strong> [[NEW_ENTRANTS_THREAT]] ([[NEW_ENTRANTS_RATIONALE]]).</li>
                <li><strong>Buyer power:</strong> [[BUYER_POWER]] due to [[BUYER_POWER_FACTORS]].</li>
                <li><strong>Supplier power:</strong> [[SUPPLIER_POWER]] ([[SUPPLIER_POWER_FACTORS]]).</li>
                <li><strong>Substitutes:</strong> [[SUBSTITUTES_THREAT]] ([[SUBSTITUTES_RATIONALE]]).</li>
                <li><strong>Rivalry:</strong> [[COMPETITIVE_RIVALRY]] ([[RIVALRY_FACTORS]]).</li>
            </ul>
            <p><strong>Implications:</strong> [[COMPETITIVE_STRATEGY_IMPLICATIONS]].<a href="#ref-23" class="citation-link">[23]</a></p>
        </section>

        <!-- 
        CORE SEGMENTS SECTION INSTRUCTIONS:
        
        1. Replace with your actual identified market segments
        2. Provide clear descriptions for each segment
        3. Include key needs and pain points
        4. Use appropriate segment CSS classes
        -->
        <section id="core-segments">
            <h2>4. Identification of Core Market Segments</h2>
            
            <h3>Market Segment Portfolio</h3>
            
            <div class="segment-grid">
                <div class="segment-card [[SEGMENT_1_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_1_NAME]]</h4>
                    <p>[[SEGMENT_1_DETAILED_DESCRIPTION]]</p>
                </div>
                <div class="segment-card [[SEGMENT_2_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_2_NAME]]</h4>
                    <p>[[SEGMENT_2_DETAILED_DESCRIPTION]]</p>
                </div>
                <div class="segment-card [[SEGMENT_3_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_3_NAME]]</h4>
                    <p>[[SEGMENT_3_DETAILED_DESCRIPTION]]</p>
                </div>
                <div class="segment-card [[SEGMENT_4_TYPE]]">
                    <h4 class="segment-title">[[SEGMENT_4_NAME]]</h4>
                    <p>[[SEGMENT_4_DETAILED_DESCRIPTION]]</p>
                </div>
            </div>

            <h3>Key Needs/Pains Snapshot</h3>
            <ul class="bullet-points">
                <li>[[KEY_NEED_1]]</li>
                <li>[[KEY_NEED_2]]</li>
                <li>[[KEY_NEED_3]]</li>
                <li>[[KEY_NEED_4]]</li>
            </ul>
        </section>

        <!-- 
        SEGMENT PROFILES SECTION INSTRUCTIONS:
        
        For each segment, provide detailed profiles including:
        1. Demographic & Firmographic Profile
        2. Psychographic & Behavioral Profile  
        3. Media Consumption & Channels
        4. Current Solution & Switching Triggers
        
        Repeat the structure below for each identified segment
        -->
        <section id="segment-profiles">
            <h2>5. Deep-Dive Segment Profiles</h2>
            
            <!-- SEGMENT A TEMPLATE -->
            <h3>Segment A: The [[SEGMENT_1_NAME]]</h3>
            <div class="sub-section">
                <h4>5.A.1 Demographic & Firmographic Profile</h4>
                <p>[[SEGMENT_1_DEMOGRAPHIC_PROFILE]]</p>
                
                <h4>5.A.2 Psychographic & Behavioral Profile</h4>
                <p>[[SEGMENT_1_PSYCHOGRAPHIC_PROFILE]]</p>
                
                <h4>5.A.3 Media Consumption & Channels</h4>
                <p>[[SEGMENT_1_MEDIA_CHANNELS]]</p>
                
                <h4>5.A.4 Current Solution & Switching Triggers</h4>
                <p>[[SEGMENT_1_SWITCHING_TRIGGERS]]</p>
            </div>

            <!-- SEGMENT B TEMPLATE -->
            <h3>Segment B: The [[SEGMENT_2_NAME]]</h3>
            <div class="sub-section">
                <h4>5.B.1 Demographic & Firmographic Profile</h4>
                <p>[[SEGMENT_2_DEMOGRAPHIC_PROFILE]]</p>
                
                <h4>5.B.2 Psychographic & Behavioral Profile</h4>
                <p>[[SEGMENT_2_PSYCHOGRAPHIC_PROFILE]]</p>
                
                <h4>5.B.3 Channels</h4>
                <p>[[SEGMENT_2_MEDIA_CHANNELS]]</p>
                
                <h4>5.B.4 Triggers</h4>
                <p>[[SEGMENT_2_SWITCHING_TRIGGERS]]</p>
            </div>

            <!-- 
            AI AGENT INSTRUCTIONS:
            Continue this pattern for Segments C, D, etc. 
            Add or remove segments based on your actual analysis
            Each segment should follow the same structure for consistency
            -->
            
            <!-- SEGMENT C TEMPLATE -->
            <h3>Segment C: The [[SEGMENT_3_NAME]]</h3>
            <div class="sub-section">
                <h4>5.C.1 Demographic & Firmographic Profile</h4>
                <p>[[SEGMENT_3_DEMOGRAPHIC_PROFILE]]</p>
                
                <h4>5.C.2 Psychographic & Behavioral Profile</h4>
                <p>[[SEGMENT_3_PSYCHOGRAPHIC_PROFILE]]</p>
                
                <h4>5.C.3 Channels</h4>
                <p>[[SEGMENT_3_MEDIA_CHANNELS]]</p>
                
                <h4>5.C.4 Triggers</h4>
                <p>[[SEGMENT_3_SWITCHING_TRIGGERS]]</p>
            </div>

            <!-- SEGMENT D TEMPLATE -->
            <h3>Segment D: The [[SEGMENT_4_NAME]]</h3>
            <div class="sub-section">
                <h4>5.D.1 Demographic & Firmographic Profile</h4>
                <p>[[SEGMENT_4_DEMOGRAPHIC_PROFILE]]</p>
                
                <h4>5.D.2 Psychographic & Behavioral Profile</h4>
                <p>[[SEGMENT_4_PSYCHOGRAPHIC_PROFILE]]</p>
                
                <h4>5.D.3 Channels</h4>
                <p>[[SEGMENT_4_MEDIA_CHANNELS]]</p>
                
                <h4>5.D.4 Triggers</h4>
                <p>[[SEGMENT_4_SWITCHING_TRIGGERS]]</p>
            </div>
        </section>

        <!-- 
        SEGMENT EVALUATION SECTION INSTRUCTIONS:
        
        1. Create attractiveness matrix based on your analysis
        2. Conduct SWOT analysis for each major segment
        3. Use appropriate tag classes (tag-high, tag-medium, tag-low)
        4. Include strategic implications for each segment
        -->
        <section id="segment-evaluation">
            <h2>6. Segment Attractiveness & Evaluation</h2>
            
            <h3>6.1 Segment Attractiveness Matrix</h3>
            
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Segment</th>
                        <th>Attractiveness</th>
                        <th>Competitive Intensity</th>
                        <th>Key Considerations</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>[[SEGMENT_1_NAME]] (A)</td>
                        <td><span class="tag [[SEGMENT_1_ATTRACTIVENESS_CLASS]]">[[SEGMENT_1_ATTRACTIVENESS]]</span></td>
                        <td><span class="tag [[SEGMENT_1_COMPETITION_CLASS]]">[[SEGMENT_1_COMPETITION]]</span></td>
                        <td>[[SEGMENT_1_KEY_CONSIDERATIONS]]</td>
                    </tr>
                    <tr>
                        <td>[[SEGMENT_2_NAME]] (B)</td>
                        <td><span class="tag [[SEGMENT_2_ATTRACTIVENESS_CLASS]]">[[SEGMENT_2_ATTRACTIVENESS]]</span></td>
                        <td><span class="tag [[SEGMENT_2_COMPETITION_CLASS]]">[[SEGMENT_2_COMPETITION]]</span></td>
                        <td>[[SEGMENT_2_KEY_CONSIDERATIONS]]</td>
                    </tr>
                    <tr>
                        <td>[[SEGMENT_3_NAME]] (C)</td>
                        <td><span class="tag [[SEGMENT_3_ATTRACTIVENESS_CLASS]]">[[SEGMENT_3_ATTRACTIVENESS]]</span></td>
                        <td><span class="tag [[SEGMENT_3_COMPETITION_CLASS]]">[[SEGMENT_3_COMPETITION]]</span></td>
                        <td>[[SEGMENT_3_KEY_CONSIDERATIONS]]</td>
                    </tr>
                    <tr>
                        <td>[[SEGMENT_4_NAME]] (D)</td>
                        <td><span class="tag [[SEGMENT_4_ATTRACTIVENESS_CLASS]]">[[SEGMENT_4_ATTRACTIVENESS]]</span></td>
                        <td><span class="tag [[SEGMENT_4_COMPETITION_CLASS]]">[[SEGMENT_4_COMPETITION]]</span></td>
                        <td>[[SEGMENT_4_KEY_CONSIDERATIONS]]</td>
                    </tr>
                </tbody>
            </table>

            <p><strong>Narrative insight:</strong> [[ATTRACTIVENESS_ANALYSIS_NARRATIVE]]<a href="#ref-24" class="citation-link">[24]</a></p>

            <!-- 
            SWOT ANALYSIS INSTRUCTIONS:
            Provide SWOT analysis for each major segment
            Use appropriate SWOT CSS classes (strengths, weaknesses, opportunities, threats)
            Include strategic implications for each segment
            -->
            <h3>6.2 Segment-Specific SWOT Analysis</h3>
            
            <h4>Segment A ([[SEGMENT_1_NAME]])</h4>
            <div class="swot-container">
                <div class="swot-card strengths">
                    <h4>Strengths</h4>
                    <ul>
                        <li>[[SEGMENT_1_STRENGTH_1]]</li>
                        <li>[[SEGMENT_1_STRENGTH_2]]</li>
                        <li>[[SEGMENT_1_STRENGTH_3]]</li>
                    </ul>
                </div>
                <div class="swot-card weaknesses">
                    <h4>Weaknesses</h4>
                    <ul>
                        <li>[[SEGMENT_1_WEAKNESS_1]]</li>
                        <li>[[SEGMENT_1_WEAKNESS_2]]</li>
                    </ul>
                </div>
                <div class="swot-card opportunities">
                    <h4>Opportunities</h4>
                    <ul>
                        <li>[[SEGMENT_1_OPPORTUNITY_1]]</li>
                        <li>[[SEGMENT_1_OPPORTUNITY_2]]</li>
                    </ul>
                </div>
                <div class="swot-card threats">
                    <h4>Threats</h4>
                    <ul>
                        <li>[[SEGMENT_1_THREAT_1]]</li>
                    </ul>
                </div>
            </div>
            <p><strong>Implications:</strong> [[SEGMENT_1_SWOT_IMPLICATIONS]]</p>

            <!-- 
            AI AGENT INSTRUCTIONS:
            Repeat the SWOT structure above for each major segment (B, C, D, etc.)
            Customize the analysis based on your specific findings
            -->

            <h4>Segment B ([[SEGMENT_2_NAME]])</h4>
            <div class="swot-container">
                <div class="swot-card strengths">
                    <h4>Strengths</h4>
                    <ul>
                        <li>[[SEGMENT_2_STRENGTH_1]]</li>
                        <li>[[SEGMENT_2_STRENGTH_2]]</li>
                        <li>[[SEGMENT_2_STRENGTH_3]]</li>
                    </ul>
                </div>
                <div class="swot-card weaknesses">
                    <h4>Weaknesses</h4>
                    <ul>
                        <li>[[SEGMENT_2_WEAKNESS_1]]</li>
                    </ul>
                </div>
                <div class="swot-card opportunities">
                    <h4>Opportunities</h4>
                    <ul>
                        <li>[[SEGMENT_2_OPPORTUNITY_1]]</li>
                        <li>[[SEGMENT_2_OPPORTUNITY_2]]</li>
                    </ul>
                </div>
                <div class="swot-card threats">
                    <h4>Threats</h4>
                    <ul>
                        <li>[[SEGMENT_2_THREAT_1]]</li>
                    </ul>
                </div>
            </div>
            <p><strong>Implications:</strong> [[SEGMENT_2_SWOT_IMPLICATIONS]]</p>

            <div class="section-highlight">
                <h4>Strategic Implications</h4>
                <p>[[OVERALL_STRATEGIC_IMPLICATIONS]]<a href="#ref-25" class="citation-link">[25]</a></p>
            </div>
        </section>

        <!-- 
        TARGETING STRATEGY SECTION INSTRUCTIONS:
        
        1. Define your recommended targeting approach
        2. Specify primary and secondary targets with rationale
        3. Include Ansoff Matrix analysis for growth options
        4. Provide strategic recommendations
        -->
        <section id="targeting-strategy">
            <h2>7. Targeting Strategy & Strategic Recommendations</h2>
            
            <div class="recommendation-box">
                <h4>Recommended Targeting Strategy</h4>
                <p>[[TARGETING_STRATEGY_RECOMMENDATION]]. This approach leverages [[STRATEGIC_RATIONALE]] to maximize ROI across segments while enabling [[IMPLEMENTATION_BENEFITS]].<a href="#ref-26" class="citation-link">[26]</a></p>
            </div>

            <h3>Recommended Primary & Secondary Targets</h3>
            <ul class="bullet-points">
                <li><strong>Primary:</strong> [[PRIMARY_TARGET]]</li>
                <li><strong>Secondary:</strong> [[SECONDARY_TARGET]]</li>
            </ul>
            <p><strong>Rationale:</strong> [[TARGETING_RATIONALE]]<a href="#ref-27" class="citation-link">[27]</a></p>

            <h3>Strategic Growth Options (Ansoff Matrix)</h3>
            <ul class="bullet-points">
                <li><strong>Market Penetration:</strong> [[MARKET_PENETRATION_STRATEGY]]</li>
                <li><strong>Product Development:</strong> [[PRODUCT_DEVELOPMENT_STRATEGY]]</li>
                <li><strong>Market Development:</strong> [[MARKET_DEVELOPMENT_STRATEGY]]</li>
                <li><strong>Diversification:</strong> [[DIVERSIFICATION_STRATEGY]]</li>
            </ul>

            <div class="section-highlight">
                <p><strong>Strategic Focus:</strong> [[ANSOFF_STRATEGIC_FOCUS]]</p>
            </div>
        </section>

        <!-- 
        POSITIONING SECTION INSTRUCTIONS:
        
        1. Develop positioning statements for primary segments
        2. Create value propositions for each segment
        3. Define messaging pillars
        4. Use positioning-card CSS class for formatted statements
        -->
        <section id="positioning">
            <h2>8. Positioning & Value Proposition Development</h2>
            
            <h3>For [[SEGMENT_1_NAME]]</h3>
            <div class="positioning-card">
                <h4>Positioning Statement</h4>
                <p class="positioning-statement">For [[SEGMENT_1_TARGET_CUSTOMER]] who [[SEGMENT_1_NEED]], [[COMPANY_NAME]] is [[SEGMENT_1_CATEGORY]] delivering [[SEGMENT_1_BENEFIT]]. Unlike [[SEGMENT_1_COMPETITION]], [[COMPANY_NAME]] provides [[SEGMENT_1_DIFFERENTIATION]].</p>
                
                <h4>Core Value Proposition</h4>
                <p>[[SEGMENT_1_VALUE_PROPOSITION]]</p>
                
                <h4>Messaging Pillars</h4>
                <ul class="bullet-points">
                    <li>[[SEGMENT_1_MESSAGE_1]]</li>
                    <li>[[SEGMENT_1_MESSAGE_2]]</li>
                    <li>[[SEGMENT_1_MESSAGE_3]]</li>
                    <li>[[SEGMENT_1_MESSAGE_4]]</li>
                </ul>
            </div>

            <h3>For [[SEGMENT_2_NAME]]</h3>
            <div class="positioning-card">
                <h4>Positioning Statement</h4>
                <p class="positioning-statement">For [[SEGMENT_2_TARGET_CUSTOMER]] seeking [[SEGMENT_2_NEED]], [[COMPANY_NAME]] offers [[SEGMENT_2_SOLUTION]] with [[SEGMENT_2_BENEFIT]]. Unlike [[SEGMENT_2_COMPETITION]], we provide [[SEGMENT_2_DIFFERENTIATION]].</p>
                
                <h4>Core Value Proposition</h4>
                <p>[[SEGMENT_2_VALUE_PROPOSITION]]</p>
                
                <h4>Messaging Pillars</h4>
                <ul class="bullet-points">
                    <li>[[SEGMENT_2_MESSAGE_1]]</li>
                    <li>[[SEGMENT_2_MESSAGE_2]]</li>
                    <li>[[SEGMENT_2_MESSAGE_3]]</li>
                    <li>[[SEGMENT_2_MESSAGE_4]]</li>
                </ul>
            </div>
        </section>

        <!-- 
        MARKETING MIX SECTION INSTRUCTIONS:
        
        1. Develop 4P strategies for primary target segments
        2. Include Product, Price, Place, and Promotion recommendations
        3. Use marketing-mix-grid and marketing-mix-card CSS classes
        4. Provide specific, actionable recommendations
        -->
        <section id="marketing-mix">
            <h2>9. Marketing Mix Implications (The 4Ps)</h2>
            
            <h3>Primary Target Segment: [[SEGMENT_1_NAME]]</h3>
            <div class="marketing-mix-grid">
                <div class="marketing-mix-card">
                    <h4>Product</h4>
                    <p>[[SEGMENT_1_PRODUCT_STRATEGY]]<a href="#ref-28" class="citation-link">[28]</a></p>
                </div>
                <div class="marketing-mix-card">
                    <h4>Price</h4>
                    <p>[[SEGMENT_1_PRICING_STRATEGY]]<a href="#ref-29" class="citation-link">[29]</a></p>
                </div>
                <div class="marketing-mix-card">
                    <h4>Place (Distribution)</h4>
                    <p>[[SEGMENT_1_DISTRIBUTION_STRATEGY]]<a href="#ref-30" class="citation-link">[30]</a></p>
                </div>
                <div class="marketing-mix-card">
                    <h4>Promotion</h4>
                    <p>[[SEGMENT_1_PROMOTION_STRATEGY]]<a href="#ref-31" class="citation-link">[31]</a></p>
                </div>
            </div>

            <h3>Secondary Target Segment: [[SEGMENT_2_NAME]]</h3>
            <div class="marketing-mix-grid">
                <div class="marketing-mix-card">
                    <h4>Product</h4>
                    <p>[[SEGMENT_2_PRODUCT_STRATEGY]]</p>
                </div>
                <div class="marketing-mix-card">
                    <h4>Price</h4>
                    <p>[[SEGMENT_2_PRICING_STRATEGY]]<a href="#ref-32" class="citation-link">[32]</a></p>
                </div>
                <div class="marketing-mix-card">
                    <h4>Place</h4>
                    <p>[[SEGMENT_2_DISTRIBUTION_STRATEGY]]</p>
                </div>
                <div class="marketing-mix-card">
                    <h4>Promotion</h4>
                    <p>[[SEGMENT_2_PROMOTION_STRATEGY]]<a href="#ref-33" class="citation-link">[33]</a></p>
                </div>
            </div>
        </section>

        <!-- 
        CONCLUSION SECTION INSTRUCTIONS:
        
        1. Synthesize key findings and strategic recommendations
        2. Highlight critical strategic choices
        3. Provide forward-looking implementation guidance
        4. Include final strategic implications
        -->
        <section id="conclusion">
            <h2>10. Conclusion: Synthesis and Forward Look</h2>
            
            <div class="key-insights">
                <h4>Synthesis</h4>
                <p>[[MARKET_SYNTHESIS]] leveraging [[COMPETITIVE_ADVANTAGE]] as a differentiator in [[COMPETITIVE_CONTEXT]]. The combination of [[PRIMARY_SEGMENTS]] provides [[STRATEGIC_VALUE]], complemented by [[SECONDARY_SEGMENTS]] as [[STRATEGIC_ROLE]].<a href="#ref-34" class="citation-link">[34]</a></p>
            </div>

            <div class="recommendation-box">
                <h4>Critical Strategic Choice</h4>
                <p>Prioritize [[STRATEGIC_PRIORITY]] as the primary segment, with [[SECONDARY_PRIORITY]] as the secondary target, while maintaining [[LONG_TERM_STRATEGY]]. The plan emphasizes [[IMPLEMENTATION_APPROACH]] to prove ROI in [[INITIAL_MARKET]] and beyond.<a href="#ref-35" class="citation-link">[35]</a></p>
            </div>

            <h3>Forward-Look</h3>
            <p>The segmentation framework supports [[RESOURCE_ALLOCATION]] and [[LEARNING_APPROACH]]. Next steps include [[NEXT_STEPS_LIST]] to maximize adoption in [[INITIAL_MARKET]] and eventually expand [[EXPANSION_STRATEGY]].<a href="#ref-36" class="citation-link">[36]</a></p>
        </section>

        <!-- 
        APPENDIX SECTION INSTRUCTIONS:
        
        1. Document methodology and data sources
        2. Include risks and mitigation strategies
        3. Provide additional context on analysis approach
        -->
        <section id="appendix">
            <h2>Sourcing & Methodology Appendix</h2>
            
            <p>Data sources include [[DATA_SOURCES_DESCRIPTION]], complemented by [[ADDITIONAL_SOURCES]]. The approach integrates [[METHODOLOGY_DESCRIPTION]] to validate segment definitions and [[VALIDATION_APPROACH]].<a href="#ref-37" class="citation-link">[37]</a></p>

            <div class="risk-warning">
                <h4>Risks and Mitigation</h4>
                <p>[[RISK_FACTORS]] with proposed mitigations including [[MITIGATION_STRATEGIES]].<a href="#ref-38" class="citation-link">[38]</a></p>
            </div>
        </section>

        <!-- 
        REFERENCES SECTION INSTRUCTIONS:
        
        CRITICAL: This is where all numbered citations link to
        
        1. Replace each [[REF_X_CONTENT]] with actual reference information
        2. Include full URLs for web sources when available
        3. Use proper academic citation format
        4. Ensure each reference has a unique ID matching the citation links above
        5. Add or remove references as needed based on your analysis
        
        CRITICAL: DO NOT CREATE OR USE CITATIONS THAT AREN'T EXPLICITLY LISTED IN THE INPUT DATA. NEVER USE "example.com" OR ANY SUCH ASSUMPTIONS.
        
        Format: [Reference Number] Source Description - "Quote or key finding" URL (if available)
        -->
        <section id="references">
            <h2>References</h2>
            
            <ul class="reference-list">
                <li id="ref-1"><span class="reference-number">[1]</span> <a href = "[[REF_1_URL]]" class="citation-link">[[REF_1_CONTENT]]</a></li>
                <li id="ref-2"><span class="reference-number">[2]</span> <a href = "[[REF_2_URL]]" class="citation-link">[[REF_2_CONTENT]]</a></li>
                <li id="ref-3"><span class="reference-number">[3]</span> <a href = "[[REF_3_URL]]" class="citation-link">[[REF_3_CONTENT]]</a></li>
                <li id="ref-4"><span class="reference-number">[4]</span> <a href = "[[REF_4_URL]]" class="citation-link">[[REF_4_CONTENT]]</a></li>
                <li id="ref-5"><span class="reference-number">[5]</span> <a href = "[[REF_5_URL]]" class="citation-link">[[REF_5_CONTENT]]</a></li>
            </ul>

        </section>

        <footer>
            <p>[Name of the product] | Date in words</p>
        </footer>
    </div>
</body>
</html>
    ```

    ---
    ### HTML TEMPLATE COMPLIANCE

    **CRITICAL:** You must use the EXACT HTML structure provided above. Do not modify the HTML structure, CSS, or JavaScript. Only replace the bracketed placeholders with actual data.

    **1. Template Sections to Fill:**
    - Replace [Product/Service Name] with actual product name
    - Replace [Report-Number] with generated report ID (format: SEG-YYYY-MM-DD-001)
    - Replace [Date of Report] with current date: {datetime.datetime.now().strftime("%B %d, %Y")}
    - Fill all bracketed placeholders throughout the document
    - Populate tables with actual data rows
    - Update KPI values and metrics
    - Replace placeholder content in each section

    **2. Data Population Rules:**
    - **Tables:** Add actual data rows inside <tbody> sections, removing placeholder rows
    - **Metrics:** Update data-value attributes and --v CSS variables for bars/charts with actual percentages
    - **Perceptual Map:** Set style="--x: [0-100]; --y: [0-100]" on .dot elements for competitor positioning
    - **Timeline:** Set style="--start: week; --end: week" on .band elements for project phases
    - **KPIs:** Replace bracketed values with actual numbers and percentages
    - **Segment Details:** Expand the details sections with complete segment profiles

    **3. Missing Data Handling:**
    - If specific data is not found, explicitly state "Information not available in research"
    - Do not leave bracketed placeholders unfilled
    - For missing metrics, use "N/A" or state "Data not found"
    - For missing sections, include a note explaining the limitation

    ---
    ### CONTENT QUALITY STANDARDS

    **1. Comprehensive Coverage:**
    - Executive Summary with key findings and TAM/SAM/SOM estimates
    - Complete PESTLE analysis for each factor with segment-specific implications
    - Detailed segment profiles with demographics, psychographics, and behaviors
    - Segment attractiveness matrix with actual scoring (0-100 scale)
    - Competitive positioning and perceptual mapping with competitor placement
    - Marketing mix implications (4Ps) for each priority segment
    - Implementation timeline with specific milestones and week numbers
    - Conclusions with actionable recommendations

    **2. Data Integration:**
    - Use actual figures from research findings
    - Include segment sizes, growth rates, and market values with proper formatting
    - Provide specific competitor names and market shares when available
    - Reference recent trends and developments (2023-2024)
    - Include geographic and demographic specifics
    - Convert percentages to --v CSS variables for visual bars (0-100 scale)

    **3. Strategic Insights:**
    - Each section should provide actionable business insights
    - Balance opportunities with challenges and risks
    - Provide clear rationale for segment prioritization
    - Connect PESTLE factors to segment implications
    - Link competitive positioning to strategic recommendations

    ---
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
    - [ ] Tables populated with real segment data (remove placeholder rows)
    - [ ] Metrics and KPIs updated with actual values
    - [ ] CSS variables set for visual elements (--v: 0-100 for bars, --x/--y: 0-100 for positioning)
    - [ ] Citations added for all factual claims
    - [ ] Missing data explicitly acknowledged where applicable
    - [ ] Professional tone maintained throughout
    - [ ] Strategic insights provided in each section

    Generate the complete HTML report using the template above with all placeholders filled with actual research data.

"""