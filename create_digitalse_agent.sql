/*
===============================================================================
DIGITALSE CORTEX AGENT SETUP
===============================================================================

This script creates the DigitalSE Cortex Agent that leverages the complete
infrastructure built by setup_digitalse.sql.

AGENT NAME: DIGITALSE
PURPOSE: Snowflake Workload Optimizer and Performance Analysis Agent

The agent provides:
1. Snowflake documentation search using Cortex Search
2. Query performance analysis using Cortex Analyst
3. Query optimization recommendations
4. Gen2 warehouse performance predictions
5. DDL/DML execution and management
6. Benchmark-based recommendations

PREREQUISITES:
- setup_digitalse.sql must be executed first
- load_benchmarks.sql should be executed for benchmark data
- ACCOUNTADMIN or appropriate role access
- Cortex AI functionality enabled

EXECUTION TIME: 2-3 minutes (includes Cortex Search service creation)

===============================================================================
*/

USE ROLE ACCOUNTADMIN;

-- ============================================================================
-- STEP 1: SET UP SNOWFLAKE DOCUMENTATION FROM MARKETPLACE
-- ============================================================================

/*
GET SNOWFLAKE DOCUMENTATION FROM MARKETPLACE
--------------------------------------------
The official Snowflake Documentation is available as a FREE listing in the 
Snowflake Marketplace.

REQUIRED STEP - Get from Marketplace:
1. Log into Snowsight UI
2. Navigate to: Data Products → Marketplace
3. Search for: "Snowflake Documentation"
4. Look for the listing by Provider: "Snowflake"
5. Click "Get" button
6. Accept the terms and mount the database
7. Database name will be: SNOWFLAKE_DOCUMENTATION

The database contains a REFERENCE schema with a DOCS table that includes:
- Product documentation (SQL, functions, features)
- Release notes and updates
- Best practices guides  
- Reference materials
- Searchable by title, content, category
- Updated regularly by Snowflake
*/

-- Verify you have the Snowflake Documentation database from marketplace
SHOW DATABASES LIKE 'SNOWFLAKE_DOCUMENTATION';

-- Expected output: Should show SNOWFLAKE_DOCUMENTATION database
-- If not shown, you MUST get it from Marketplace first (see instructions above)

-- Verify access to the documentation schema
SHOW SCHEMAS IN DATABASE SNOWFLAKE_DOCUMENTATION;

-- For shared databases, use GRANT IMPORTED PRIVILEGES instead of individual grants
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE_DOCUMENTATION TO ROLE DIGITALSE_ADMIN_RL;

-- Verify the search service was created successfully
SHOW CORTEX SEARCH SERVICES IN SCHEMA SNOWFLAKE_DOCUMENTATION.SHARED;

-- Test the search service with a sample query
SELECT 
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'SNOWFLAKE_DOCUMENTATION.SHARED.CKE_SNOWFLAKE_DOCS_SERVICE',
        '{
            "query": "clustering keys",
            "columns": ["CHUNK", "DOCUMENT_TITLE", "SOURCE_URL"],
            "limit": 3
        }'
    ) AS search_results;

SELECT 
    '✅ Snowflake Documentation Search Service Created!' AS status,
    'SNOWFLAKE_DOCUMENTATION.SHARED.CKE_SNOWFLAKE_DOCS_SERVICE' AS service_name,
    'Ready for agent integration' AS next_step;

-- ============================================================================
-- STEP 3: CREATE THE DIGITALSE CORTEX AGENT
-- ============================================================================

USE DATABASE SNOWFLAKE_INTELLIGENCE;
USE SCHEMA AGENTS;

-- Create the DigitalSE Cortex Agent
CREATE OR REPLACE AGENT DIGITALSE
  COMMENT = 'DigitalSE - AI-powered Snowflake query performance analysis and optimization agent'
  PROFILE = '{"display_name": "DigitalSE", "avatar": "analytics-icon.png", "color": "#29B5E8"}'
  FROM SPECIFICATION
$$
models:
  orchestration: claude-4-sonnet

orchestration:
  budget:
    seconds: 60
    tokens: 32000

instructions:
  response: |
    You are DigitalSE, a friendly and expert Snowflake Workload Optimizer.
   
    Communication Style:
    - Be conversational, helpful, and encouraging
    - Explain technical concepts clearly without overwhelming jargon
    - Provide actionable recommendations with clear next steps
    - Use structured formatting (bullet points, numbered lists) for clarity
    - Always explain WHY a recommendation matters, not just WHAT to do
   
    Response Structure:
    1. Acknowledge the user's question/concern
    2. Provide analysis with key findings
    3. Offer specific, prioritized recommendations
    4. Suggest next steps or follow-up questions
   
  orchestration: |
    Tool Selection Guidelines:
   
    1. For questions about Snowflake features, syntax, best practices, or documentation:
       - Use "SNOWFLAKE_DOCUMENTATION" to search official Snowflake documentation
       - Examples: "How does clustering work?", "What is dynamic table syntax?",
                   "Best practices for semi-structured data"
   
    2. For questions about query history, performance metrics, or workload patterns:
       - Use "AccountUsageAnalyst" to query ACCOUNT_USAGE data
   
    3. For analyzing specific query performance (when user provides query_id):
       - Use "QueryDataFetcher" to get detailed operator statistics
       - Use "QueryTextFetcher" to retrieve query text and metadata
       - Use "FieldDefinitions" to understand query metrics
       - Use "AnalysisGuidance" for performance threshold interpretation
   
    4. For Gen2 warehouse performance predictions:
       - Use "Gen2BenchmarkLookup" with query description to get improvement estimates
   
    5. For DDL operations (CREATE, ALTER, DROP, SHOW, DESCRIBE):
       - Use "GetObjectDDL" to extract DDL for objects
       - Use "ExecuteDDL" to execute DDL statements
   
    6. For DML operations (SELECT, INSERT, UPDATE, DELETE):
       - Use "ExecuteDML" to execute DML statements
   
    Workflow Strategy:
    - For concept questions, start with SNOWFLAKE_DOCUMENTATION to provide authoritative context
    - For workload analysis, use AccountUsageAnalyst for broad patterns
    - Drill down to specific queries using QueryDataFetcher
    - Cross-reference with benchmarks using Gen2BenchmarkLookup
    - Combine documentation best practices with actual workload data
    - Provide comprehensive recommendations with both theory and practical insights
   
  system: |
    You are DigitalSE, an expert Snowflake Workload Optimizer designed to help users:
   
    Core Capabilities:
    1. Analyze query performance and identify bottlenecks
    2. Recommend warehouse sizing and Gen2 warehouse benefits
    3. Identify opportunities for clustering keys, search optimization, and materialized views
    4. Detect expensive query patterns (exploding joins, spilling, inefficient scans)
    5. Provide cost optimization recommendations
    6. Execute DDL/DML operations safely
    7. Generate benchmark-based performance predictions
   
    Analysis Framework:
    - Query Execution: Analyze operator statistics, execution time breakdown, I/O patterns
    - Pruning Efficiency: Evaluate partition and micro-partition pruning effectiveness
    - Cache Utilization: Assess cache hit rates and remote I/O costs
    - Resource Usage: Identify spilling, memory pressure, network transfer bottlenecks
    - Cost Attribution: Connect performance to credit consumption
   
    Optimization Focus Areas:
    1. **Gen2 Warehouses**: Recommend Gen2 for SCANS, JOINS, DML, SEMI-STRUCTURED data operations
    2. **Clustering**: Identify tables with poor pruning that need clustering keys
    3. **Search Optimization**: Detect point lookup and selective filter patterns
    4. **Materialized Views**: Find repeated aggregation patterns
    5. **Query Acceleration**: Identify queries with outlier portions eligible for QAS
    6. **Warehouse Sizing**: Right-size warehouses based on query load and concurrency
   
    Always provide context-aware recommendations based on actual workload patterns.

  sample_questions:
    - question: "How do clustering keys work in Snowflake?"
      answer: "Let me search the Snowflake documentation to explain clustering keys and their benefits for query performance."
   
    - question: "What queries consumed the most credits last week?"
      answer: "I'll analyze your query history to identify the top credit-consuming queries and provide optimization recommendations."
   
    - question: "Which tables would benefit from clustering keys?"
      answer: "Let me examine your table pruning history to identify tables with poor partition pruning that would benefit from clustering keys."
   
    - question: "Show me queries eligible for query acceleration service (QAS)"
      answer: "I'll query the query acceleration eligible data to find queries that would benefit from QAS and estimate potential improvements."
   
    - question: "How much faster would this query run on a Gen2 warehouse?"
      answer: "I'll analyze your query pattern and compare it to Gen2 benchmarks to estimate performance improvements."
   
    - question: "Analyze query performance for query ID abc123"
      answer: "I'll fetch detailed operator statistics for that query and provide a comprehensive performance analysis with optimization recommendations."
   
    - question: "What are my most expensive warehouses this month?"
      answer: "Let me analyze warehouse usage and cost attribution to identify your highest credit-consuming warehouses."
   
    - question: "Find queries with spilling or memory pressure"
      answer: "I'll search for queries that spilled to local or remote storage and recommend warehouse sizing or query optimizations."
   
    - question: "Which queries would benefit most from materialized views?"
      answer: "I'll analyze repeated aggregation patterns in your query history to identify candidates for materialized views."
   
    - question: "Show me tables with low cache hit rates"
      answer: "Let me examine table scanning patterns to find tables that frequently cause remote I/O and might need optimization."
   
    - question: "What is the DDL for table MY_TABLE?"
      answer: "I'll extract the complete DDL definition for that table using the DDL extractor tool."

tools:
  - tool_spec:
      type: "cortex_analyst_text_to_sql"
      name: "AccountUsageAnalyst"
      description: |
        Cortex Analyst tool for analyzing Snowflake ACCOUNT_USAGE data.
       
        Use this tool to:
        - Query query history, attribution, and insights
        - Analyze warehouse load and query acceleration eligibility
        - Examine partition pruning patterns and table/column access
        - Aggregate query metrics (credits, execution time, bytes scanned)
        - Identify expensive queries, users, warehouses, or tables
        - Find queries eligible for QAS or Gen2 warehouses
       
        The semantic view includes 8 joined tables:
        1. QUERY_HISTORY - Complete query execution history
        2. QUERY_ATTRIBUTION_HISTORY - Credit consumption by query
        3. QUERY_INSIGHTS - AI-generated query optimization insights
        4. QUERY_ACCELERATION_ELIGIBLE - QAS eligibility data
        5. TABLE_QUERY_PRUNING_HISTORY - Table-level pruning metrics
        6. COLUMN_QUERY_PRUNING_HISTORY - Column-level pruning metrics
        7. TABLE_PRUNING_HISTORY - Overall table pruning efficiency
        8. WAREHOUSE_LOAD_HISTORY - Warehouse concurrency and queuing
       
        Available time range: Last 365 days (with 45-minute latency)
 
  - tool_spec:
      type: "cortex_search"
      name: "SNOWFLAKE_DOCUMENTATION"
      description: |
        Searches the official Snowflake documentation using Cortex Search.
       
        Use this tool to find information about:
        - Snowflake features and capabilities
        - SQL syntax and functions
        - Best practices and recommendations
        - Configuration options and parameters
        - Optimization techniques
        - New features and releases
        - Security and governance
        - Data sharing and collaboration
       
        This tool provides authoritative, up-to-date information from
        Snowflake's official documentation to supplement your analysis
        and recommendations.
       
        Use this when users ask:
        - "How does [feature] work?"
        - "What is the syntax for [command]?"
        - "Best practices for [topic]"
        - "Tell me about [Snowflake feature]"
        - "What's new in [version/feature]?"
 
  - tool_spec:
      type: "generic"
      name: "QueryDataFetcher"
      description: |
        Fetches detailed query operator statistics for a specific query ID.
       
        Use this tool to analyze:
        - Query execution plan and operator tree
        - Operator-level statistics (rows, bytes, execution time)
        - I/O patterns (cache hit rates, bytes scanned)
        - Partition pruning efficiency
        - Spilling detection (local/remote)
        - Join explosion detection
        - Bottleneck identification
       
        Input: query_id (UUID format), output_format ('pretty' or 'minified')
        Output: JSON with summary metrics and detailed operator statistics
       
        Call this when user asks to "analyze query", "why is my query slow",
        or provides a specific query ID for investigation.
      input_schema:
        type: "object"
        properties:
          query_id:
            type: "string"
            description: "The query ID to analyze (UUID format)"
          output_format:
            type: "string"
            description: "Output format: 'pretty' or 'minified'"
        required:
          - "query_id"
 
  - tool_spec:
      type: "generic"
      name: "QueryTextFetcher"
      description: |
        Retrieves query text and execution metadata for a specific query ID.
       
        Returns:
        - Complete SQL query text (formatted)
        - User and role information
        - Warehouse size used
        - Query load percentage
       
        Input: query_id (UUID format)
        Output: JSON with query details
       
        Use this to see the actual SQL for a query before analyzing performance.
      input_schema:
        type: "object"
        properties:
          query_id:
            type: "string"
            description: "The query ID to retrieve (UUID format)"
        required:
          - "query_id"
 
  - tool_spec:
      type: "generic"
      name: "FieldDefinitions"
      description: |
        Returns comprehensive field definitions for Snowflake query operator statistics.
       
        Use this reference to understand:
        - OPERATOR_STATISTICS fields (I/O, pruning, spilling, DML)
        - EXECUTION_TIME_BREAKDOWN components
        - Core operator fields (OPERATOR_TYPE, PARENT_OPERATORS)
       
        Call this when you need to explain query metrics to the user.
      input_schema:
        type: "object"
        properties: {}
 
  - tool_spec:
      type: "generic"
      name: "AnalysisGuidance"
      description: |
        Returns performance threshold guidance for query analysis.
       
        Provides benchmarks for:
        - Execution time thresholds (GREAT/GOOD/POOR/CRITICAL)
        - Table scan efficiency criteria
        - Join performance (row multiplication factors)
        - Spilling severity levels
       
        Use this to categorize query performance and set expectations.
      input_schema:
        type: "object"
        properties: {}
 
  - tool_spec:
      type: "generic"
      name: "Gen2BenchmarkLookup"
      description: |
        Looks up Gen2 warehouse performance improvement estimate based on query pattern. It returns the percentage improvement from Gen1 to Gen2. A negative number indicates a performance improvement and a positive number indicates a performance degradation.
       
        Uses AI to match query description to benchmark data for:
        - Table scans
        - Joins
        - DML operations
        - Semi-structured data queries
        - Window functions
       
        Input: QUERY_DESC_INPUT (description of query operations)
        Output: Estimated speedup factor (AWS Gen2 vs Gen1 average)
       
        Use this to predict Gen2 warehouse benefits for specific queries.
      input_schema:
        type: "object"
        properties:
          QUERY_DESC_INPUT:
            type: "string"
            description: "Description of query operations (e.g., 'large table scan', 'complex joins')"
        required:
          - "QUERY_DESC_INPUT"
 
  - tool_spec:
      type: "generic"
      name: "GetObjectDDL"
      description: |
        Extracts DDL for any Snowflake object.
       
        Supported object types: TABLE, VIEW, SCHEMA, DATABASE, FUNCTION,
        PROCEDURE, WAREHOUSE, DYNAMIC_TABLE, STREAM, TASK, PIPE, and more.
       
        Inputs:
        - object_type: Type of object (e.g., 'TABLE', 'VIEW')
        - object_name: Fully qualified name (e.g., 'DB.SCHEMA.TABLE')
        - use_fully_qualified_names: Boolean (default TRUE)
       
        Output: Formatted DDL with metadata analysis
       
        Use this when user asks to "show DDL", "get table definition",
        or "what is the structure of".
      input_schema:
        type: "object"
        properties:
          object_type:
            type: "string"
            description: "Type of object (TABLE, VIEW, FUNCTION, etc.)"
          object_name:
            type: "string"
            description: "Fully qualified object name"
          use_fully_qualified_names:
            type: "boolean"
            description: "Use fully qualified names in DDL output"
        required:
          - "object_type"
          - "object_name"
 
  - tool_spec:
      type: "generic"
      name: "ExecuteDDL"
      description: |
        Executes DDL statements (CREATE, ALTER, DROP, SHOW, DESCRIBE, GRANT, etc.).
       
        Inputs:
        - ddl_statement: SQL DDL statement to execute
        - output_format: 'pretty', 'minified', or 'table' (default 'pretty')
       
        Output: JSON with execution status and results
       
        ⚠️ SAFETY: Always confirm with user before executing DROP or destructive DDL.
        Use this for object creation, modification, or information retrieval.
      input_schema:
        type: "object"
        properties:
          ddl_statement:
            type: "string"
            description: "DDL statement to execute"
          output_format:
            type: "string"
            description: "Output format: 'pretty', 'minified', or 'table'"
        required:
          - "ddl_statement"
 
  - tool_spec:
      type: "generic"
      name: "ExecuteDML"
      description: |
        Executes DML statements (SELECT, INSERT, UPDATE, DELETE, MERGE).
       
        Inputs:
        - dml_statement: SQL DML statement to execute
        - output_format: 'table', 'json', or 'summary' (default 'table')
       
        Output: Query results formatted per output_format
       
        ⚠️ SAFETY: Be cautious with UPDATE/DELETE. Recommend WHERE clauses.
        Use this for data queries and modifications.
      input_schema:
        type: "object"
        properties:
          dml_statement:
            type: "string"
            description: "DML statement to execute"
          output_format:
            type: "string"
            description: "Output format: 'table', 'json', or 'summary'"
        required:
          - "dml_statement"

tool_resources:
  AccountUsageAnalyst:
    semantic_view: "DIGITALSE.PUBLIC.ACCOUNT_USAGE_SEMANTIC_VIEW"
 
  SNOWFLAKE_DOCUMENTATION:
    name: "SNOWFLAKE_DOCUMENTATION.SHARED.CKE_SNOWFLAKE_DOCS_SERVICE"
    max_results: 5
 
  QueryDataFetcher:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.QUERY_DEMO.QUERY_DATA_FETCHER"
 
  QueryTextFetcher:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.TOOLS.QUERY_TEXT"
 
  FieldDefinitions:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.QUERY_DEMO.FIELD_DEFINITIONS"
 
  AnalysisGuidance:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.QUERY_DEMO.ANALYSIS_GUIDANCE"
 
  Gen2BenchmarkLookup:
    type: "function"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.BENCHMARK.FN_BENCHMARK_LOOKUP_BY_QUERY"
 
  GetObjectDDL:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.TOOLS.GET_OBJECT_DDL"
 
  ExecuteDDL:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.TOOLS.EXECUTE_DDL"
 
  ExecuteDML:
    type: "procedure"
    execution_environment:
      type: "warehouse"
      warehouse: "DIGITALSE_WH"
    identifier: "DIGITALSE.TOOLS.EXECUTE_DML"
$$;



-- Grant necessary permissions for the agent
GRANT USAGE ON DATABASE DIGITALSE TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON SCHEMA DIGITALSE.TOOLS TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON SCHEMA DIGITALSE.QUERY_DEMO TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON SCHEMA DIGITALSE.BENCHMARK TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON SCHEMA DIGITALSE.PUBLIC TO ROLE DIGITALSE_ADMIN_RL;

-- Grant execute permissions on procedures and functions
GRANT USAGE ON PROCEDURE DIGITALSE.QUERY_DEMO.QUERY_DATA_FETCHER(STRING, STRING) TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON PROCEDURE DIGITALSE.TOOLS.QUERY_TEXT(VARCHAR) TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON PROCEDURE DIGITALSE.QUERY_DEMO.FIELD_DEFINITIONS() TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON PROCEDURE DIGITALSE.QUERY_DEMO.ANALYSIS_GUIDANCE() TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON PROCEDURE DIGITALSE.TOOLS.GET_OBJECT_DDL(STRING, STRING, BOOLEAN) TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON PROCEDURE DIGITALSE.TOOLS.EXECUTE_DDL(STRING, STRING) TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON PROCEDURE DIGITALSE.TOOLS.EXECUTE_DML(STRING, STRING) TO ROLE DIGITALSE_ADMIN_RL;
GRANT USAGE ON FUNCTION DIGITALSE.BENCHMARK.FN_BENCHMARK_LOOKUP_BY_QUERY(VARCHAR) TO ROLE DIGITALSE_ADMIN_RL;

-- Grant select on semantic view
GRANT SELECT ON DIGITALSE.PUBLIC.ACCOUNT_USAGE_SEMANTIC_VIEW TO ROLE DIGITALSE_ADMIN_RL;

-- Grant access to Snowflake Documentation (imported/shared database from Marketplace)


-- Grant warehouse usage
GRANT USAGE ON WAREHOUSE DIGITALSE_WH TO ROLE DIGITALSE_ADMIN_RL;

-- ============================================================================
-- SETUP COMPLETE - ALL COMPONENTS READY
-- ============================================================================

SELECT 
    '🎉 DIGITALSE CORTEX AGENT SETUP COMPLETE!' AS status,
    CURRENT_TIMESTAMP() AS completion_time;

SELECT 
    'Component' AS component_type,
    'Status' AS status,
    'Location' AS location
UNION ALL
SELECT 
    '📚 Documentation Search Service',
    '✅ Created',
    'SNOWFLAKE_DOCUMENTATION.SHARED.CKE_SNOWFLAKE_DOCS_SERVICE'
UNION ALL
SELECT 
    '🤖 DigitalSE Agent',
    '✅ Created',
    'DIGITALSE.AGENTS.DIGITALSE'
UNION ALL
SELECT 
    '🔧 All Tools Integrated',
    '✅ Ready',
    '9 tools configured'
UNION ALL
SELECT
    '🔐 Permissions',
    '✅ Granted',
    'DIGITALSE_ADMIN_RL has full access';

-- Verify all components
SHOW CORTEX SEARCH SERVICES IN SCHEMA SNOWFLAKE_DOCUMENTATION.SHARED;
SHOW AGENTS IN SCHEMA DIGITALSE.AGENTS;

-- Example: How to use the agent
/*
-- Ask about Snowflake features using documentation search
SELECT SNOWFLAKE.CORTEX.COMPLETE_AGENT(
    'DIGITALSE.AGENTS.DIGITALSE',
    'How do clustering keys work in Snowflake and when should I use them?'
) AS agent_response;

-- Start a conversation with the agent about workload analysis
SELECT SNOWFLAKE.CORTEX.COMPLETE_AGENT(
    'DIGITALSE.AGENTS.DIGITALSE',
    'What queries consumed the most credits last week?'
) AS agent_response;

-- Analyze a specific query
SELECT SNOWFLAKE.CORTEX.COMPLETE_AGENT(
    'DIGITALSE.AGENTS.DIGITALSE',
    'Analyze query performance for query ID: 01234567-89ab-cdef-0123-456789abcdef'
) AS agent_response;

-- Get Gen2 recommendations
SELECT SNOWFLAKE.CORTEX.COMPLETE_AGENT(
    'DIGITALSE.AGENTS.DIGITALSE',
    'Which of my queries would benefit most from Gen2 warehouses?'
) AS agent_response;

-- Combined documentation and analysis
SELECT SNOWFLAKE.CORTEX.COMPLETE_AGENT(
    'DIGITALSE.AGENTS.DIGITALSE',
    'Explain search optimization and tell me which of my tables would benefit from it'
) AS agent_response;
*/

-- ============================================================================
-- TROUBLESHOOTING GUIDE
-- ============================================================================

/*
ISSUE: Cannot find SNOWFLAKE_DOCUMENTATION database
----------------------------------------------------
SOLUTION:
1. Check if you have ACCOUNTADMIN role:
   USE ROLE ACCOUNTADMIN;
   
2. Get from Snowflake Marketplace (REQUIRED):
   - Go to Snowsight UI → Data Products → Marketplace
   - Search: "Snowflake Documentation"
   - Provider: Snowflake (official listing, FREE)
   - Click "Get" button to mount the shared database
   - This will create SNOWFLAKE_DOCUMENTATION database
   
3. Verify it was added:
   SHOW DATABASES LIKE 'SNOWFLAKE_DOCUMENTATION';
   
4. If still not available, contact Snowflake Support

ISSUE: Cortex Search Service creation fails
--------------------------------------------
SOLUTION:
1. Verify Cortex AI is enabled in your account (contact Snowflake if not)

2. Ensure DIGITALSE_WH warehouse exists and is available:
   SHOW WAREHOUSES LIKE 'DIGITALSE_WH';
   
3. Check permissions:
   USE ROLE ACCOUNTADMIN;
   GRANT CREATE CORTEX SEARCH SERVICE ON SCHEMA DIGITALSE.CORTEX_SEARCH TO ROLE ACCOUNTADMIN;

4. Verify SNOWFLAKE_DOCUMENTATION database exists and has IMPORTED PRIVILEGES:
   SHOW DATABASES LIKE 'SNOWFLAKE_DOCUMENTATION';
   SHOW GRANTS ON DATABASE SNOWFLAKE_DOCUMENTATION;
   
5. Ensure you can read from the shared documentation table:
   SELECT COUNT(*) FROM SNOWFLAKE_DOCUMENTATION.REFERENCE.DOCS;
   
6. Try creating with a different warehouse if DIGITALSE_WH is not available

ISSUE: Agent creation fails with "tool not found" error
--------------------------------------------------------
SOLUTION:
1. Ensure all prerequisites are completed:
   - Run setup_digitalse.sql first
   - Run load_benchmarks.sql for benchmark data
   - Verify all procedures exist:
     SHOW PROCEDURES IN DIGITALSE.TOOLS;
     SHOW PROCEDURES IN DIGITALSE.QUERY_DEMO;
     SHOW FUNCTIONS IN DIGITALSE.BENCHMARK;

2. Verify semantic view exists:
   SHOW VIEWS IN DIGITALSE.PUBLIC LIKE 'ACCOUNT_USAGE_SEMANTIC_VIEW';

3. Check all permissions are granted to DIGITALSE_ADMIN_RL

ISSUE: Agent responds but doesn't use documentation search
-----------------------------------------------------------
SOLUTION:
1. Test the search service directly:
   SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
       DIGITALSE.CORTEX_SEARCH.DOCUMENT_SEARCH_SERVICE,
       '{"query": "clustering", "columns": ["TITLE", "CONTENT"], "limit": 3}'
   );

2. Verify permissions on the search service:
   GRANT READ ON CORTEX SEARCH SERVICE DIGITALSE.CORTEX_SEARCH.DOCUMENT_SEARCH_SERVICE 
       TO ROLE DIGITALSE_ADMIN_RL;

3. Verify permissions on the shared documentation database:
   GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE_DOCUMENTATION TO ROLE DIGITALSE_ADMIN_RL;

4. Explicitly ask documentation questions:
   "Search the documentation for clustering keys"
   "What does the official documentation say about search optimization?"

ISSUE: Query analysis tools return errors
------------------------------------------
SOLUTION:
1. Check ACCOUNT_USAGE access:
   SELECT COUNT(*) FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY LIMIT 1;
   
2. Note: ACCOUNT_USAGE has 45-minute latency. Recent queries may not be available.

3. Verify query IDs are valid UUIDs from your account

ADDITIONAL RESOURCES:
---------------------
- Snowflake Documentation: https://docs.snowflake.com
- Cortex AI Documentation: https://docs.snowflake.com/en/user-guide/snowflake-cortex
- Cortex Search: https://docs.snowflake.com/en/user-guide/cortex-search
- Cortex Agents: https://docs.snowflake.com/en/user-guide/cortex-agents

For support with this DigitalSE setup, review:
- setup_digitalse.sql - Main infrastructure setup
- load_benchmarks.sql - Benchmark data setup
- create_digitalse_agent.sql (this file) - Agent configuration
*/

