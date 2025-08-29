

TEMPLATE = """You are an expert internal management consultant and organizational analyst. Your mission is to synthesize all available information into a single, comprehensive report that provides a clear-eyed assessment of the organization's internal health, operational capabilities, and strategic positioning.

OBJECTIVE: Create a detailed HTML report that serves as a mirror for the organization, reflecting its true strengths, weaknesses, and potential. This is for internal strategic planning and decision-making, not for external sales.

CRITICAL INSTRUCTIONS:

OMIT ANY SECTION OR SUBSECTION that cannot be populated with the current findings. Do not include empty sections or placeholder text.

Provide the complete, styled HTML structure as your output.

REPLACE ALL placeholder text [like this] with the synthesized research and analysis.

Maintain all styling classes and structure for proper rendering.

The analysis should be frank, evidence-based, and actionable.

Length Guidance: For sections requesting "between 2 and 4 paragraphs," provide substantive, insightful analysis. A "paragraph" is considered 3-5 concise, information-dense sentences. Do not be overly verbose; prioritize clarity and impact.

EXPECTED OUTPUT: A complete, standalone HTML document following the structure below, with all placeholders replaced by substantive organizational analysis content.

```html
html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Internal Organizational Capability Report</title>
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
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 0.9em;
        }
        
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
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="report-header">
            <h1>Internal Organizational Capability Report</h1>
            <div class="report-subtitle">A Strategic Analysis of Strengths, Weaknesses, and Opportunities</div>
        </div>

        <div class="report-meta">
            <div>
                <strong>Report Generated:</strong> <span>[Current Date]</span>
            </div>
            <div>
                <strong>Subject:</strong> <span>[Organization Name] - Internal Analysis</span>
            </div>
            <div>
                <strong>Confidentiality:</strong> <span>Strictly Internal Use</span>
            </div>
        </div>

        <div class="table-of-contents">
            <h2>Table of Contents</h2>
            <ul class="toc-list">
                <li><a href="#executive-summary">1. Executive Summary</a></li>
                <li><a href="#core-capabilities">2. Core Capabilities & Operational Model</a></li>
                <li><a href="#financial-health">3. Financial Health & Resource Allocation</a></li>
                <li><a href="#human-capital">4. Human Capital & Leadership Analysis</a></li>
                <li><a href="#tech-ops">5. Technology & Operational Infrastructure</a></li>
                <li><a href="#strategic-position">6. Strategic Market Position</a></li>
                <li><a href="#cultural-assessment">7. Cultural Assessment & Organizational Health</a></li>
                <li><a href="#swot">8. SWOT Analysis</a></li>
                <li><a href="#strategic-imperatives">9. Strategic Imperatives & Recommendations</a></li>
                <li><a href="#appendices">10. Appendices</a></li>
            </ul>
        </div>

        <!-- SECTION 1: EXECUTIVE SUMMARY -->
        <div class="executive-summary">
            <h2 id="executive-summary">1. Executive Summary</h2>
            <div class="content-section">
                <h3>Organizational Profile</h3>
                <div>
                    <strong>Organization Name:</strong> [Company legal name, DBA names]<br>
                    <strong>Core Mission:</strong> [Statement of purpose]<br>
                    <strong>Operational Status:</strong> [e.g., Growth, Maturity, Transformation, Turnaround]<br>
                    <strong>Key Metrics:</strong> [Employee count, core operational units, geographic footprint]
                </div>

                <h3>Overall Health Assessment</h3>
                <div class="sub-section">
                    <p><strong>Operational Effectiveness:</strong></p>
                    <div>
                        [Provide a 2-4 paragraph synthesis of how efficiently the organization converts inputs into outputs. Assess process maturity, productivity metrics, and quality standards. Conclude with an overall verdict on operational health.]
                    </div>

                    <p><strong>Financial Vitality:</strong></p>
                    <div>
                        [Provide a 2-4 paragraph high-level assessment of financial stability, profitability, investment capacity, and sustainability of the current business model. Highlight key strengths and concerns.]
                    </div>

                    <p><strong>Strategic Coherence:</strong></p>
                    <div>
                        [Provide a 2-4 paragraph analysis of how well the organization's structure, culture, capabilities, and strategy are aligned towards common goals. Identify any major misalignments.]
                    </div>
                </div>

                <h3>Critical Insights</h3>
                <div class="key-findings">
                    <div class="finding-card critical">
                        <h4>Primary Vulnerabilities</h4>
                        <div>
                            [Write 1-2 paragraphs summarizing the most significant internal risks and weaknesses that require urgent attention. Be direct and evidence-based.]
                        </div>
                    </div>
                    <div class="finding-card opportunity">
                        <h4>Core Strengths & Unrealized Potentials</h4>
                        <div>
                            [Write 1-2 paragraphs summarizing the organization's strongest assets and the most promising opportunities for leverage and growth that are not yet fully capitalized upon.]
                        </div>
                    </div>
                    <div class="finding-card">
                        <h4>Strategic Implications</h4>
                        <div>
                            [Write 1-2 paragraphs on what this overall assessment means for the future direction and priorities of the organization. What is the overarching narrative?]
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION 2: CORE CAPABILITIES -->
        <h2 id="core-capabilities">2. Core Capabilities & Operational Model</h2>
        <div class="content-section">
            <h3>Value Creation Engine</h3>
            <div class="sub-section">
                <p><strong>How We Create Value:</strong></p>
                <div>
                    [Provide a 2-4 paragraph detailed analysis of the organization's primary methods for creating value for its customers and stakeholders. Describe the key activities and processes that define the business.]
                </div>
                <p><strong>Core Competencies:</strong></p>
                <div>
                    [Provide a 2-4 paragraph identification and analysis of what the organization does uniquely better than anyone else. These are the roots of competitive advantage. Be specific and justify each claimed competency.]
                </div>
            </div>

            <h3>Operational Structure & Processes</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph analysis of the organizational design (e.g., functional, matrix, product-based). Analyze key workflows, decision-rights frameworks, and process efficiency. Identify any structural bottlenecks or advantages.]
                </div>
            </div>
        </div>

        <!-- SECTION 3: FINANCIAL HEALTH -->
        <h2 id="financial-health">3. Financial Health & Resource Allocation</h2>
        <div class="content-section">
            <h3>Efficiency & Investment Analysis</h3>
            <div class="sub-section">
                <p><strong>Spending Patterns & ROI:</strong></p>
                <div>
                    [Provide a 2-4 paragraph analysis of capital and operational expenditure. Where does the money go? What is the return on key investments? Assess cost structure and efficiency compared to industry norms.]
                </div>
                <p><strong>Resource Allocation Strategy:</strong></p>
                <div>
                    [Provide a 2-4 paragraph evaluation of how well financial and human resources are allocated to strategic priorities. Are resources focused on the most important areas for growth and sustainability?]
                </div>
            </div>

            <h3>Financial Metrics Overview</h3>
            <div class="data-grid">
                <div class="data-card">
                    <h4>Revenue Stability</h4>
                    <div class="metric-value">[Diversity Score, e.g., High/Med/Low or %]</div>
                    <div class="metric-label">Diversity & Recurrence of Revenue</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 75%;"></div>
                    </div>
                </div>
                <div class="data-card">
                    <h4>Profitability Efficiency</h4>
                    <div class="metric-value">[e.g., 15% Margin]</div>
                    <div class="metric-label">Margin Analysis vs. Benchmark</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 60%;"></div>
                    </div>
                </div>
                <div class="data-card">
                    <h4>Investment Capacity</h4>
                    <div class="metric-value">[e.g., $XXM Cash]</div>
                    <div class="metric-label">Cash Flow & Liquidity</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 85%;"></div>
                    </div>
                </div>
                <div class="data-card">
                    <h4>Cost Management</h4>
                    <div class="metric-value">[Efficiency Ratio]</div>
                    <div class="metric-label">Operational Efficiency</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 70%;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION 4: HUMAN CAPITAL -->
        <h2 id="human-capital">4. Human Capital & Leadership Analysis</h2>
        <div class="content-section">
            <h3>Talent Landscape</h3>
            <div class="sub-section">
                <p><strong>Workforce Composition:</strong></p>
                <div>
                    [Provide a 2-4 paragraph analysis of skills distribution, experience levels, diversity, and critical talent roles. Identify skills gaps and surpluses. Comment on the overall quality and depth of talent.]
                </div>
                <p><strong>Culture & Engagement:</strong></p>
                <div>
                    [Provide a 2-4 paragraph assessment of employee morale, engagement scores, retention rates, and the health of the organizational culture. What are the key cultural drivers?]
                </div>
            </div>

            <h3>Leadership Effectiveness</h3>
            <div class="sub-section">
                <p><strong>Strategic Alignment:</strong></p>
                <div>
                    [Provide a 2-4 paragraph analysis of how well leadership communicates and drives the strategy throughout the organization. Is there a unified vision?]
                </div>
                <p><strong>Decision-Making Efficacy:</strong></p>
                <div>
                    [Provide a 2-4 paragraph analysis of the speed, quality, and inclusiveness of decision-making processes. Are decisions made at the right level?]
                </div>
            </div>
        </div>

        <!-- SECTION 5: TECHNOLOGY & OPERATIONS -->
        <h2 id="tech-ops">5. Technology & Operational Infrastructure</h2>
        <div class="content-section">
            <h3>Technology Stack Maturity</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph assessment of the current technology's ability to support business goals. Evaluate scalability, security, integration, and modernness. Identify technical debt and its implications.]
                </div>
            </div>

            <h3>Operational Resilience</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph analysis of the robustness of key operational systems (production, logistics, customer service, etc.). Assess capacity, single points of failure, bottlenecks, and overall risk exposure.]
                </div>
            </div>
        </div>

        <!-- SECTION 6: STRATEGIC POSITION -->
        <h2 id="strategic-position">6. Strategic Market Position</h2>
        <div class="content-section">
            <h3>Competitive Advantage</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph frank assessment of the durability and sources of the organization's competitive advantage (e.g., Cost, Differentiation, Focus). How strong and defensible is this advantage?]
                </div>
            </div>

            <h3>Brand & Reputation Equity</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph analysis of the strength of the brand, customer loyalty, market perception, and overall reputation as a key intangible asset. How does this impact operational effectiveness?]
                </div>
            </div>
        </div>

        <!-- SECTION 7: CULTURAL ASSESSMENT -->
        <h2 id="cultural-assessment">7. Cultural Assessment & Organizational Health</h2>
        <div class="content-section">
            <h3>Core Cultural Traits</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph description of the observable culture: values, behaviors, rituals, and unwritten rules that define "how things are done around here." Is the culture an asset or a liability?]
                </div>
            </div>

            <h3>Adaptability & Learning</h3>
            <div class="sub-section">
                <div>
                    [Provide a 2-4 paragraph assessment of the organization's ability to learn, innovate, and adapt to change. Is the culture a catalyst or an impediment to change? Assess the tolerance for risk and failure.]
                </div>
            </div>
        </div>

        <!-- SECTION 8: SWOT -->
        <h2 id="swot">8. SWOT Analysis</h2>
        <div class="content-section">
            <div class="data-grid" style="grid-template-columns: 1fr 1fr; gap: 15px;">
                <div class="data-card" style="background-color: #f0fff4; text-align: left;">
                    <h4>Strengths</h4>
                    <ul class="bullet-points">
                        <li>[Internal capability that provides an advantage]</li>
                        <li>[Unique resource or asset]</li>
                        <li>[Strong brand reputation]</li>
                        <li>[Patents, intellectual property]</li>
                        <li>[Expertise, knowledge, data]</li>
                    </ul>
                </div>
                <div class="data-card" style="background-color: #fff5f5; text-align: left;">
                    <h4>Weaknesses</h4>
                    <ul class="bullet-points">
                        <li>[Internal limitation or gap]</li>
                        <li>[Area where competitors are stronger]</li>
                        <li>[Operational inefficiency]</li>
                        <li>[Financial constraint]</li>
                        <li>[Reputational challenge]</li>
                    </ul>
                </div>
                <div class="data-card" style="background-color: #e8f6ff; text-align: left;">
                    <h4>Opportunities</h4>
                    <ul class="bullet-points">
                        <li>[Favorable market or industry trend]</li>
                        <li>[New technology to leverage]</li>
                        <li>[Change in customer behavior]</li>
                        <li>[Potential partnership or alliance]</li>
                        <li>[New market segment to enter]</li>
                    </ul>
                </div>
                <div class="data-card" style="background-color: #fff4e0; text-align: left;">
                    <h4>Threats</h4>
                    <ul class="bullet-points">
                        <li>[Emerging competitor or disruptive innovation]</li>
                        <li>[Changing regulatory landscape]</li>
                        <li>[Negative economic shift]</li>
                        <li>[Substitute product/service]</li>
                        <li>[Key talent shortage]</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- SECTION 9: STRATEGIC IMPERATIVES -->
        <h2 id="strategic-imperatives">9. Strategic Imperatives & Recommendations</h2>
        <div class="content-section">
            <h3>Priority Initiatives</h3>
            <div class="sub-section">
                <div class="recommendation-box">
                    <h4>Strengthen Core Capabilities</h4>
                    <div>
                        [Provide 2-4 paragraphs of actionable recommendations for investing in and bolstering the organization's key strengths. Be specific about what actions to take and why.]
                    </div>
                </div>
                <div class="recommendation-box">
                    <h4>Address Critical Vulnerabilities</h4>
                    <div>
                        [Provide 2-4 paragraphs of an actionable plan for mitigating the most severe weaknesses and threats identified in the analysis. Prioritize based on impact and urgency.]
                    </div>
                </div>
                <div class="recommendation-box">
                    <h4>Capitalize on Key Opportunities</h4>
                    <div>
                        [Provide 2-4 paragraphs of a actionable strategy for leveraging strengths to exploit the most promising opportunities. Define what success looks like.]
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION 10: APPENDICES -->
        <h2 id="appendices">10. Appendices</h2>
        <div class="content-section">
            <h3>Appendix A: Key Data Summary</h3>
            <p><em>[If applicable, include summarized data tables here]</em></p>

            <h3>Appendix B: Research Sources</h3>
            <p><em>[List the key sources of information used to compile this report]</em></p>
        </div>

        <footer>
            <p>© [Current Year] - Internal Organizational Analysis Report. Confidential and Proprietary.</p>
        </footer>
    </div>
</body>
</html>
```

    **CRITICAL INSTRUCTIONS:**
    - Provide the complete HTML structure above as your output
    - REPLACE ALL placeholder text exactly as shown
    - Include all sections unless specifically excluded in the research plan
    - Maintain the styling classes and structure for proper rendering
    - The researcher will replace placeholders with actual research findings
    """