# Digital SE

**AI-Powered Snowflake Workload Optimizer and Performance Analysis Platform**

---

## Overview

Digital SE is an intelligent Snowflake-based platform that leverages Snowflake Cortex AI to provide comprehensive query performance analysis, optimization recommendations, and workload management capabilities. At its core is the **DigitalSE Cortex Agent**, a conversational AI assistant powered by Claude 4 Sonnet that helps Snowflake users optimize their data workloads, reduce costs, and improve query performance.

## Purpose

Digital SE serves as your expert companion for:

- **Query Performance Analysis**: Deep-dive into query execution plans, operator statistics, and performance bottlenecks
- **Cost Optimization**: Identify credit-consuming queries and recommend warehouse sizing optimizations
- **Gen2 Warehouse Assessment**: Predict performance improvements by migrating to Gen2 warehouses
- **Intelligent Recommendations**: Get AI-driven suggestions for clustering keys, search optimization, and materialized views
- **Documentation Integration**: Access official Snowflake documentation contextually within your analysis
- **DDL/DML Management**: Safely execute and manage database operations with built-in tools

## Key Features

### 🔍 Comprehensive Query Analysis
- Operator-level performance statistics
- Execution time breakdown analysis
- I/O pattern detection (cache hit rates, remote I/O)
- Partition pruning efficiency evaluation
- Spilling detection (local and remote storage)
- Join explosion identification

### 📊 Workload Intelligence
- Query history analysis with 365-day retention
- Credit attribution and cost tracking
- Warehouse load and concurrency patterns
- Query acceleration service (QAS) eligibility detection
- Table and column-level pruning insights

### 🚀 Optimization Recommendations
- **Gen2 Warehouses**: Performance improvement predictions based on benchmarks
- **Clustering Keys**: Identify tables with poor pruning efficiency
- **Search Optimization**: Detect point lookup and selective filter patterns
- **Materialized Views**: Find repeated aggregation opportunities
- **Query Acceleration**: Identify outlier queries eligible for QAS
- **Warehouse Sizing**: Right-size warehouses based on workload patterns

### 📚 Integrated Documentation Search
- Cortex Search-powered access to official Snowflake documentation
- Contextual documentation retrieval during analysis
- Best practices and syntax reference

### 🛠️ Advanced Tools
- Universal DDL extractor supporting 20+ object types
- Safe DML/DDL execution environment
- Query history tracking and retrieval
- Benchmark-based performance predictions
- AI-generated catalog descriptions

## Architecture

### Components

Digital SE consists of two main setup scripts:

#### 1. Infrastructure Setup (`setup_digitalse.sql`)
Creates the complete platform infrastructure:

- **Core Infrastructure**
  - Dedicated `DIGITALSE_ADMIN_RL` role
  - Optimized `DIGITALSE_WH` warehouse (X-SMALL, auto-suspend)
  - `DIGITALSE` database with organized schemas

- **Schemas**
  - `TOOLS`: Custom functions and stored procedures
  - `AGENTS`: AI agent definitions
  - `INTEGRATIONS`: External service configurations
  - `QUERY_DEMO`: Query analysis utilities
  - `BENCHMARK`: Performance benchmark data and functions
  - `HORIZON_CATALOG`: AI-generated object descriptions

- **Tools & Procedures**
  - `GET_OBJECT_DDL`: Universal DDL extraction
  - `EXECUTE_DDL`: Safe DDL execution
  - `EXECUTE_DML`: DML execution with safeguards
  - `QUERY_DATA_FETCHER`: Detailed query statistics retrieval
  - `QUERY_TEXT_FETCHER`: Query text and metadata extraction
  - `FIELD_DEFINITIONS`: Query metric documentation
  - `ANALYSIS_GUIDANCE`: Performance threshold benchmarks
  - `FN_BENCHMARK_LOOKUP_BY_QUERY`: Gen2 improvement predictions

- **Semantic Views**
  - Comprehensive `ACCOUNT_USAGE_SEMANTIC_VIEW` integrating:
    - Query history (365 days)
    - Query attribution and credits
    - Query insights (AI-generated)
    - Query acceleration eligibility
    - Table/column pruning history
    - Warehouse load patterns

#### 2. Agent Creation (`create_digitalse_agent.sql`)
Configures the DigitalSE Cortex Agent with:

- **AI Model**: Claude 4 Sonnet (orchestration)
- **Budget**: 60-second, 32K token limit per interaction
- **Tool Integration**: 9 specialized tools
- **Documentation Search**: Snowflake Marketplace documentation via Cortex Search
- **Cortex Analyst**: Text-to-SQL for ACCOUNT_USAGE queries

## Prerequisites

- **Snowflake Account**: Enterprise or Business Critical edition recommended
- **Role Access**: ACCOUNTADMIN or equivalent privileges
- **Cortex AI**: Enabled in your Snowflake account
- **Snowflake Documentation**: From Snowflake Marketplace (free listing)
- **CORTEX_USER Role**: Database role for Cortex functionality

## Installation

### Step 1: Run Infrastructure Setup
```sql
-- Execute setup_digitalse.sql
-- Estimated time: 5-10 minutes
-- Creates database, schemas, procedures, functions, and semantic views
```

### Step 2: Get Snowflake Documentation
1. Navigate to **Snowsight UI → Data Products → Marketplace**
2. Search for "Snowflake Documentation"
3. Provider: **Snowflake** (official, FREE)
4. Click **Get** to mount the database
5. Database will be named: `SNOWFLAKE_DOCUMENTATION`

### Step 3: Create the DigitalSE Agent
```sql
-- Execute create_digitalse_agent.sql
-- Estimated time: 2-3 minutes
-- Creates the Cortex Agent with all tool integrations
```

## Usage Examples
Log into Snowflake Intelligence select the DitialSE agent and ask questions like: 
### Analyze Query Performance
    'Analyze query performance for query ID: 01234567-89ab-cdef-0123-456789abcdef'

### Find Top Credit Consumers
    'What queries consumed the most credits last week?'

### Get Clustering Key Recommendations
    'Which tables would benefit from clustering keys?'


### Learn About Snowflake Features
    'How do clustering keys work in Snowflake and when should I use them?'


### Gen2 Warehouse Assessment
    'Which of my queries would benefit most from Gen2 warehouses?'


### Combined Analysis and Documentation
    'Explain search optimization and tell me which of my tables would benefit from it'


## Agent Capabilities

### 9 Integrated Tools

1. **AccountUsageAnalyst** (Cortex Analyst)
   - Text-to-SQL queries against ACCOUNT_USAGE
   - Query history, attribution, and insights
   - Warehouse load and QAS eligibility
   - Table/column pruning patterns

2. **SNOWFLAKE_DOCUMENTATION** (Cortex Search)
   - Official Snowflake documentation search
   - Features, syntax, and best practices
   - Configuration and optimization guidance

3. **QueryDataFetcher** (Procedure)
   - Detailed operator statistics
   - Execution plans and bottlenecks
   - I/O patterns and spilling detection

4. **QueryTextFetcher** (Procedure)
   - SQL query text retrieval
   - Execution metadata and context

5. **FieldDefinitions** (Procedure)
   - Operator statistics field reference
   - Metric explanations

6. **AnalysisGuidance** (Procedure)
   - Performance threshold benchmarks
   - Severity categorization

7. **Gen2BenchmarkLookup** (Function)
   - AI-powered benchmark matching
   - Gen2 performance predictions

8. **GetObjectDDL** (Procedure)
   - Universal DDL extraction
   - 20+ object types supported

9. **ExecuteDDL** (Procedure)
   - Safe DDL execution
   - CREATE, ALTER, DROP, SHOW, DESCRIBE

10. **ExecuteDML** (Procedure)
    - DML query execution
    - SELECT, INSERT, UPDATE, DELETE, MERGE

## Sample Questions

The DigitalSE agent can answer questions like:

- "How do clustering keys work in Snowflake?"
- "What queries consumed the most credits last week?"
- "Which tables would benefit from clustering keys?"
- "Show me queries eligible for query acceleration service (QAS)"
- "How much faster would this query run on a Gen2 warehouse?"
- "Analyze query performance for query ID abc123"
- "What are my most expensive warehouses this month?"
- "Find queries with spilling or memory pressure"
- "Which queries would benefit most from materialized views?"
- "Show me tables with low cache hit rates"
- "What is the DDL for table MY_TABLE?"

## Communication Style

DigitalSE is designed to be:

- **Conversational**: Friendly and approachable tone
- **Educational**: Explains WHY, not just WHAT
- **Actionable**: Provides specific, prioritized recommendations
- **Structured**: Uses bullet points and numbered lists for clarity
- **Comprehensive**: Combines documentation with real workload insights

### Response Structure
1. Acknowledge the user's question/concern
2. Provide analysis with key findings
3. Offer specific, prioritized recommendations
4. Suggest next steps or follow-up questions

## Data Retention

- **Query History**: 365 days (via ACCOUNT_USAGE)
- **Account Usage Latency**: ~45 minutes
- **Semantic View**: Real-time access to account usage data
- **Benchmark Data**: Static reference data (updated via load script)

## Performance Thresholds

The agent uses standardized performance benchmarks:

- **Execution Time**:
  - GREAT: < 1 second
  - GOOD: 1-10 seconds
  - POOR: 10-60 seconds
  - CRITICAL: > 60 seconds

- **Table Scan Efficiency**: Pruning ratio analysis
- **Join Performance**: Row multiplication factors
- **Spilling Severity**: Local vs. remote storage spilling

## Optimization Focus Areas

1. **Gen2 Warehouses**: Scan, join, DML, and semi-structured query improvements
2. **Clustering**: Poor pruning patterns requiring clustering keys
3. **Search Optimization**: Point lookups and selective filters
4. **Materialized Views**: Repeated aggregation patterns
5. **Query Acceleration**: Outlier portions eligible for QAS
6. **Warehouse Sizing**: Concurrency and workload-based recommendations

## Security & Permissions

- **Dedicated Role**: `DIGITALSE_ADMIN_RL` with controlled permissions
- **Role Hierarchy**: Granted to SYSADMIN for proper governance
- **Warehouse Control**: Dedicated `DIGITALSE_WH` with auto-suspend
- **Safe Execution**: DDL/DML procedures with safeguards
- **Account Usage Access**: Read-only access to performance data

## Troubleshooting

### Documentation Database Not Found
1. Verify ACCOUNTADMIN role access
2. Get from Snowflake Marketplace (Data Products → Marketplace)
3. Search: "Snowflake Documentation" by Snowflake (FREE)
4. Mount the shared database

### Agent Creation Fails
1. Ensure `setup_digitalse.sql` was executed first
2. Verify all procedures and functions exist
3. Check semantic view creation
4. Confirm CORTEX_USER role is granted

### Documentation Search Not Working
1. Test Cortex Search service directly
2. Verify IMPORTED PRIVILEGES on SNOWFLAKE_DOCUMENTATION
3. Check search service permissions

### Recent Queries Not Available
- ACCOUNT_USAGE has ~45-minute latency
- Very recent queries may not be visible yet

## Estimated Costs

- **Warehouse**: X-SMALL with auto-suspend (300 seconds idle)
- **Cortex AI**: Pay-per-token usage (Claude 4 Sonnet)
- **Cortex Search**: Minimal compute costs
- **Storage**: Minimal (primarily metadata and benchmarks)

**Cost Control**:
- Budget limits: 60 seconds, 32K tokens per agent interaction
- Auto-suspend warehouse configuration
- Efficient query execution patterns

## Support & Resources

- **Snowflake Documentation**: https://docs.snowflake.com
- **Cortex AI**: https://docs.snowflake.com/en/user-guide/snowflake-cortex
- **Cortex Search**: https://docs.snowflake.com/en/user-guide/cortex-search
- **Cortex Agents**: https://docs.snowflake.com/en/user-guide/cortex-agents

## Files in This Repository

- `setup_digitalse.sql`: Complete infrastructure setup (5-10 min execution)
- `create_digitalse_agent.sql`: Agent creation and configuration (2-3 min execution)
- `README.md`: This documentation file

---

**Digital SE** - Intelligent Snowflake Performance Analysis, Powered by AI 🚀

