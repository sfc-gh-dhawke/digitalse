/*
===============================================================================
DIGITALSE - COMPLETE SETUP SCRIPT
===============================================================================

This script sets up the complete DigitalSE infrastructure in a single database
called DIGITALSE, combining multiple components:

1. Core Infrastructure (Role, Warehouse, Database, Schemas)
2. Object Management Tools (DDL/DML Extractors and Managers)
3. Query Analysis Tools (Data Fetcher, History, Field Definitions)
4. Prompting Tools (Analysis Guidance, Query Optimization Workflow)
5. Semantic Views (Account Usage Views)
6. Benchmark Functions
7. Horizon Catalog (AI-Generated Descriptions)

PREREQUISITES:
- ACCOUNTADMIN role access
- Ability to create roles, warehouses, databases, and schemas
- Snowflake Cortex AI functionality enabled
- SNOWFLAKE.CORTEX_USER database role

EXECUTION TIME: 5-10 minutes

TABLE OF CONTENTS (Search for these markers):
- [SECTION_1_INFRASTRUCTURE]    : Core setup (role, warehouse, database, schemas)
- [SECTION_2_OBJECT_MANAGEMENT] : DDL/DML tools
- [SECTION_3_QUERY_TOOLS]       : Query analysis and history tools
- [SECTION_4_PROMPTING_TOOLS]   : AI prompting and optimization
- [SECTION_5_SEMANTIC_VIEWS]    : Account usage semantic views
- [SECTION_6_BENCHMARK]         : Benchmark functions
- [SECTION_7_HORIZON_CATALOG]   : AI catalog management

===============================================================================
*/

-- ============================================================================
-- [SECTION_1_INFRASTRUCTURE] CORE INFRASTRUCTURE SETUP
-- ============================================================================

USE ROLE ACCOUNTADMIN;

-- Create dedicated admin role for DigitalSE
CREATE OR REPLACE ROLE DIGITALSE_ADMIN_RL 
    COMMENT = 'Administrative role for DigitalSE platform with full access to AI/ML resources';

-- Create optimized warehouse for AI workloads
USE ROLE SYSADMIN;

CREATE OR REPLACE WAREHOUSE DIGITALSE_WH
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300              -- Suspend after 5 minutes of inactivity
    AUTO_RESUME = TRUE             -- Auto-resume on query execution
    MIN_CLUSTER_COUNT = 1
    MAX_CLUSTER_COUNT = 1
    SCALING_POLICY = 'STANDARD'
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Dedicated warehouse for DigitalSE AI workloads, Cortex Search, and query execution';

-- Create main database
CREATE OR REPLACE DATABASE DIGITALSE
    COMMENT = 'Main database for DigitalSE platform containing agents, tools, integrations, and semantic views';

-- Create organized schemas
CREATE OR REPLACE SCHEMA DIGITALSE.TOOLS
    COMMENT = 'Schema for custom tools which are functions and stored procedures';

CREATE OR REPLACE SCHEMA DIGITALSE.AGENTS
    COMMENT = 'Schema for AI agents';

CREATE OR REPLACE SCHEMA DIGITALSE.INTEGRATIONS  
    COMMENT = 'Schema for external service integrations, secrets, and API configurations';

CREATE OR REPLACE SCHEMA DIGITALSE.QUERY_DEMO
    COMMENT = 'Schema for query analysis and optimization tools';

CREATE OR REPLACE SCHEMA DIGITALSE.HORIZON_CATALOG
    COMMENT = 'Schema for Horizon Catalog AI-generated descriptions';

CREATE OR REPLACE SCHEMA DIGITALSE.BENCHMARK
    COMMENT = 'Schema for benchmark data and functions';

-- Grant the new role to the current user automatically
DECLARE
    SQL_COMMAND STRING;
BEGIN
    SQL_COMMAND := 'GRANT ROLE DIGITALSE_ADMIN_RL TO USER "' || CURRENT_USER() || '";';
    EXECUTE IMMEDIATE SQL_COMMAND;
    RETURN 'Role DIGITALSE_ADMIN_RL granted successfully to user ' || CURRENT_USER();
END;

-- Transfer ownership of all created objects to the admin role
USE ROLE ACCOUNTADMIN;
GRANT OWNERSHIP ON DATABASE DIGITALSE TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON SCHEMA DIGITALSE.TOOLS TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON SCHEMA DIGITALSE.AGENTS TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON SCHEMA DIGITALSE.INTEGRATIONS TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON SCHEMA DIGITALSE.QUERY_DEMO TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON SCHEMA DIGITALSE.HORIZON_CATALOG TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON SCHEMA DIGITALSE.BENCHMARK TO ROLE DIGITALSE_ADMIN_RL;
GRANT OWNERSHIP ON WAREHOUSE DIGITALSE_WH TO ROLE DIGITALSE_ADMIN_RL;

-- Grant the admin role to SYSADMIN for role hierarchy
GRANT ROLE DIGITALSE_ADMIN_RL TO ROLE SYSADMIN;

-- Grant Cortex user role
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE DIGITALSE_ADMIN_RL;

-- ============================================================================
-- [SECTION_2_OBJECT_MANAGEMENT] DDL/DML TOOLS
-- ============================================================================

USE SCHEMA DIGITALSE.TOOLS;

/*
===============================================================================
UNIVERSAL DDL EXTRACTOR - COMPREHENSIVE DDL RETRIEVAL TOOL
===============================================================================
*/

CREATE OR REPLACE PROCEDURE GET_OBJECT_DDL(
    object_type STRING,
    object_name STRING,
    use_fully_qualified_names BOOLEAN DEFAULT TRUE
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'get_object_ddl'
COMMENT = 'Universal DDL extractor for any supported Snowflake object type'
AS
$$
import re

def get_object_ddl(session, object_type, object_name, use_fully_qualified_names):
    # Define supported object types with metadata
    supported_types = {
        'ALERT': {'recursive': False, 'requires_namespace': True},
        'DATABASE': {'recursive': True, 'requires_namespace': False},
        'DATA_METRIC_FUNCTION': {'recursive': False, 'requires_namespace': True},
        'CONTACT': {'recursive': False, 'requires_namespace': True},
        'DBT_PROJECT': {'recursive': False, 'requires_namespace': True},
        'DYNAMIC_TABLE': {'recursive': False, 'requires_namespace': True},
        'EVENT_TABLE': {'recursive': False, 'requires_namespace': True},
        'EXTERNAL_TABLE': {'recursive': False, 'requires_namespace': True},
        'FILE_FORMAT': {'recursive': False, 'requires_namespace': True},
        'HYBRID_TABLE': {'recursive': False, 'requires_namespace': True},
        'ICEBERG_TABLE': {'recursive': False, 'requires_namespace': True},
        'PIPE': {'recursive': False, 'requires_namespace': True},
        'POLICY': {'recursive': False, 'requires_namespace': True},
        'SCHEMA': {'recursive': True, 'requires_namespace': True},
        'SEMANTIC_VIEW': {'recursive': False, 'requires_namespace': True},
        'SEQUENCE': {'recursive': False, 'requires_namespace': True},
        'STORAGE_INTEGRATION': {'recursive': False, 'requires_namespace': False},
        'PROCEDURE': {'recursive': False, 'requires_namespace': True},
        'STREAM': {'recursive': False, 'requires_namespace': True},
        'TABLE': {'recursive': False, 'requires_namespace': True},
        'TAG': {'recursive': False, 'requires_namespace': True},
        'TASK': {'recursive': False, 'requires_namespace': True},
        'FUNCTION': {'recursive': False, 'requires_namespace': True},
        'VIEW': {'recursive': False, 'requires_namespace': True},
        'WAREHOUSE': {'recursive': False, 'requires_namespace': False}
    }

    try:
        # Normalize object type to uppercase
        obj_type = object_type.upper()

        # Validate object type
        if obj_type not in supported_types:
            return f"Error: Unsupported object type '{object_type}'. Supported types: {', '.join(sorted(supported_types.keys()))}"

        # Build GET_DDL query
        if use_fully_qualified_names:
            ddl_query = f"SELECT GET_DDL('{obj_type}', '{object_name}', TRUE)"
        else:
            ddl_query = f"SELECT GET_DDL('{obj_type}', '{object_name}')"

        # Execute query
        result = session.sql(ddl_query).collect()

        if not result:
            return f"No DDL found for {obj_type}: {object_name}"

        ddl_statement = result[0][0]

        # Build output
        output = f"DDL for {obj_type}: {object_name}\n"
        output += "=" * 80 + "\n\n"

        # Format the DDL for better readability
        if ddl_statement:
            # Clean up the DDL formatting
            formatted_ddl = ddl_statement.replace('\\n', '\n').replace('\\t', '    ')
            output += "DDL STATEMENT:\n"
            output += "-" * 40 + "\n"
            output += formatted_ddl + "\n\n"

            # Add metadata analysis
            output += "DDL ANALYSIS:\n"
            output += "-" * 40 + "\n"
            output += f"Object Type: {obj_type}\n"
            output += f"Object Name: {object_name}\n"
            output += f"Fully Qualified Names: {'YES' if use_fully_qualified_names else 'NO'}\n"

            # Type-specific analysis
            if obj_type == 'DYNAMIC_TABLE' and "TARGET_LAG" in ddl_statement:
                lag_match = re.search(r"TARGET_LAG\s*=\s*'([^']+)'", ddl_statement)
                if lag_match:
                    output += f"Target Lag: {lag_match.group(1)}\n"

            if "WAREHOUSE" in ddl_statement:
                warehouse_match = re.search(r"WAREHOUSE\s*=\s*(\w+)", ddl_statement)
                if warehouse_match:
                    output += f"Warehouse: {warehouse_match.group(1)}\n"

            if obj_type in ['TABLE', 'VIEW', 'DYNAMIC_TABLE']:
                if "AS SELECT" in ddl_statement or "AS\nSELECT" in ddl_statement:
                    output += "Contains SQL Query: YES\n"

            if obj_type in ['FUNCTION', 'PROCEDURE']:
                # Check for language
                lang_match = re.search(r"LANGUAGE\s+(\w+)", ddl_statement)
                if lang_match:
                    output += f"Language: {lang_match.group(1)}\n"

                # Check for runtime version
                runtime_match = re.search(r"RUNTIME_VERSION\s*=\s*'([^']+)'", ddl_statement)
                if runtime_match:
                    output += f"Runtime Version: {runtime_match.group(1)}\n"

            # Check if recursive
            if supported_types[obj_type]['recursive']:
                output += f"Recursive DDL: YES (includes all child objects)\n"

            # Count lines in DDL
            ddl_lines = ddl_statement.count('\n') + 1
            output += f"DDL Statement Lines: {ddl_lines}\n"
        else:
            output += "No DDL statement returned\n"

        return output

    except Exception as e:
        error_msg = str(e)
        if "does not exist or not authorized" in error_msg:
            return f"Error: Object {obj_type} '{object_name}' does not exist or you lack permissions to view it."
        elif "Invalid object type" in error_msg:
            return f"Error: '{object_type}' is not a valid object type for GET_DDL."
        else:
            return f"Error extracting DDL for {object_type} '{object_name}': {error_msg}"
$$;

-- Helper function to list all supported DDL types
CREATE OR REPLACE FUNCTION LIST_SUPPORTED_DDL_TYPES()
RETURNS TABLE (
    object_type STRING,
    is_recursive BOOLEAN,
    requires_namespace BOOLEAN,
    notes STRING
)
LANGUAGE SQL
COMMENT = 'Lists all object types supported by GET_DDL with their characteristics'
AS
$$
SELECT * FROM VALUES
    ('ALERT', FALSE, TRUE, 'Alerts for monitoring and notifications'),
    ('DATABASE', TRUE, FALSE, 'Databases - recursive, includes all child objects'),
    ('DATA_METRIC_FUNCTION', FALSE, TRUE, 'Data metric functions'),
    ('CONTACT', FALSE, TRUE, 'Notification contacts'),
    ('DBT_PROJECT', FALSE, TRUE, 'dbt project objects'),
    ('DYNAMIC_TABLE', FALSE, TRUE, 'Dynamic tables with automatic refresh'),
    ('EVENT_TABLE', FALSE, TRUE, 'Event tables for logging'),
    ('EXTERNAL_TABLE', FALSE, TRUE, 'External tables pointing to cloud storage'),
    ('FILE_FORMAT', FALSE, TRUE, 'File format definitions'),
    ('HYBRID_TABLE', FALSE, TRUE, 'Hybrid tables'),
    ('ICEBERG_TABLE', FALSE, TRUE, 'Apache Iceberg tables'),
    ('PIPE', FALSE, TRUE, 'Pipes for continuous data loading'),
    ('POLICY', FALSE, TRUE, 'Various policy types (masking, row access, etc.)'),
    ('SCHEMA', TRUE, TRUE, 'Schemas - recursive, includes all child objects'),
    ('SEMANTIC_VIEW', FALSE, TRUE, 'Semantic views - requires REFERENCES privilege'),
    ('SEQUENCE', FALSE, TRUE, 'Sequences for generating unique numbers'),
    ('STORAGE_INTEGRATION', FALSE, FALSE, 'Storage integrations for cloud access'),
    ('PROCEDURE', FALSE, TRUE, 'Stored procedures - include arg types for overloaded'),
    ('STREAM', FALSE, TRUE, 'Streams for change data capture'),
    ('TABLE', FALSE, TRUE, 'Regular tables - interchangeable with VIEW'),
    ('TAG', FALSE, TRUE, 'Tags for object classification'),
    ('TASK', FALSE, TRUE, 'Tasks for scheduled operations'),
    ('FUNCTION', FALSE, TRUE, 'User-defined functions - include arg types for overloaded'),
    ('VIEW', FALSE, TRUE, 'Views including materialized - interchangeable with TABLE'),
    ('WAREHOUSE', FALSE, FALSE, 'Virtual warehouses for compute')
    AS t(object_type, is_recursive, requires_namespace, notes)
$$;

/*
===============================================================================
DDL MANAGER - EXECUTE DDL STATEMENTS
===============================================================================
*/

CREATE OR REPLACE PROCEDURE EXECUTE_DDL(
    ddl_statement STRING,
    output_format STRING DEFAULT 'pretty'
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
AS
$$
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

def format_table_output(df, columns: List[str], max_rows: int = 100) -> str:
    """Format dataframe as ASCII table with proper column alignment."""
    if len(df) == 0:
        return "No objects found."
    
    # Calculate column widths
    col_widths = []
    for i, col in enumerate(columns):
        max_width = len(col)
        for row in df[:min(max_rows, len(df))]:
            val_len = len(str(row[i]) if row[i] is not None else 'NULL')
            if val_len > max_width:
                max_width = val_len
        col_widths.append(min(max_width, 40))
    
    # Build table
    output = ""
    
    # Header
    header = '| '
    separator = '|-'
    for i, col in enumerate(columns):
        header += col[:col_widths[i]].ljust(col_widths[i]) + ' | '
        separator += '-' * col_widths[i] + '-|-'
    
    output += header + '\n'
    output += separator + '\n'
    
    # Data rows
    for idx, row in enumerate(df):
        if idx >= max_rows:
            output += f"\n... {len(df) - max_rows} more rows ..."
            break
        row_str = '| '
        for i, val in enumerate(row):
            val_str = str(val) if val is not None else 'NULL'
            val_str = val_str[:col_widths[i]]
            row_str += val_str.ljust(col_widths[i]) + ' | '
        output += row_str + '\n'
    
    return output

def process_show_results(df, columns: List[str]) -> List[Dict[str, Any]]:
    """Convert SHOW/DESCRIBE results to JSON-friendly format."""
    results = []
    for row in df:
        row_dict = {}
        for i, col in enumerate(columns):
            val = row[i]
            # Convert to JSON-serializable types
            if val is None:
                row_dict[col] = None
            elif isinstance(val, (int, float, bool)):
                row_dict[col] = val
            else:
                row_dict[col] = str(val)
        results.append(row_dict)
    return results

def main(session, ddl_statement: str, output_format: str = 'pretty') -> str:
    """Main handler function for executing DDL statements."""
    try:
        # Validate input
        if not ddl_statement or not ddl_statement.strip():
            return json.dumps({
                "status": "error",
                "error": "DDL statement cannot be empty",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2 if output_format == 'pretty' else None)
        
        # Get statement type
        statement_parts = ddl_statement.strip().split()
        if not statement_parts:
            return json.dumps({
                "status": "error",
                "error": "Invalid DDL statement",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2 if output_format == 'pretty' else None)
        
        statement_type = statement_parts[0].upper()
        
        # Common DDL keywords
        ddl_keywords = ['CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'GRANT', 'REVOKE', 
                       'SHOW', 'DESCRIBE', 'DESC', 'USE', 'COMMENT']
        
        if statement_type not in ddl_keywords:
            return json.dumps({
                "status": "error",
                "error": f"Expected DDL statement, got: {statement_type}",
                "statement": ddl_statement,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2 if output_format == 'pretty' else None)
        
        # Execute the DDL
        result = session.sql(ddl_statement)
        
        # Handle SHOW and DESCRIBE commands differently
        if statement_type in ['SHOW', 'DESCRIBE', 'DESC']:
            df = result.collect()
            columns = result.schema.names
            
            # For table format, return ASCII table directly
            if output_format == 'table':
                output = f"Statement: {ddl_statement}\n"
                output += f"Results: {len(df)} rows\n\n"
                output += format_table_output(df, columns)
                return output
            
            # For JSON formats, build structured response
            response = {
                "status": "success",
                "statement_type": statement_type,
                "statement": ddl_statement,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": {
                    "row_count": len(df),
                    "columns": columns,
                    "data": process_show_results(df, columns)
                }
            }
            
            # Return based on format preference
            if output_format == 'minified':
                return json.dumps(response, separators=(',', ':'))
            else:
                return json.dumps(response, indent=2)
            
        else:
            # For other DDL, execute and return success
            result.collect()
            
            response = {
                "status": "success",
                "statement_type": statement_type,
                "statement": ddl_statement,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"DDL executed successfully"
            }
            
            # Add additional info for specific operations
            if statement_type == 'CREATE':
                if 'TABLE' in ddl_statement.upper():
                    response["object_type"] = "TABLE"
                elif 'VIEW' in ddl_statement.upper():
                    response["object_type"] = "VIEW"
                elif 'SCHEMA' in ddl_statement.upper():
                    response["object_type"] = "SCHEMA"
                elif 'DATABASE' in ddl_statement.upper():
                    response["object_type"] = "DATABASE"
            elif statement_type == 'DROP':
                response["operation"] = "DROP"
            elif statement_type == 'ALTER':
                response["operation"] = "ALTER"
            
            # Return based on format preference
            if output_format == 'minified':
                return json.dumps(response, separators=(',', ':'))
            else:
                return json.dumps(response, indent=2)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        response = {
            "status": "error",
            "error": str(e),
            "details": error_details.replace('\n', ' | '),
            "statement": ddl_statement,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Return based on format preference
        if output_format == 'minified':
            return json.dumps(response, separators=(',', ':'))
        else:
            return json.dumps(response, indent=2)
$$;

/*
===============================================================================
DML MANAGER - EXECUTE DML STATEMENTS
===============================================================================
*/

CREATE OR REPLACE PROCEDURE EXECUTE_DML(
    dml_statement STRING,
    output_format STRING DEFAULT 'table'  -- 'table', 'json', 'summary'
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'execute_dml'
AS
$$
import json

def execute_dml(session, dml_statement, output_format):
    try:
        # Validate input
        if not dml_statement or not dml_statement.strip():
            return "Error: DML statement cannot be empty"
        
        # Get statement type
        statement_type = dml_statement.strip().split()[0].upper()
        
        # Validate it's a DML statement
        if statement_type not in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE']:
            return f"Error: Only DML statements are allowed. Got: {statement_type}"
        
        # Execute the statement
        result = session.sql(dml_statement)
        
        if statement_type == 'SELECT':
            # For SELECT, collect and format results
            df = result.collect()
            columns = result.schema.names
            
            if output_format == 'json':
                rows = []
                for row in df:
                    row_dict = {columns[i]: str(row[i]) for i in range(len(columns))}
                    rows.append(row_dict)
                return json.dumps({
                    "statement": dml_statement,
                    "type": "SELECT",
                    "row_count": len(df),
                    "data": rows
                }, indent=2)
            
            elif output_format == 'summary':
                return f"SELECT executed successfully. Retrieved {len(df)} rows."
            
            else:  # table format
                output = f"Statement: {dml_statement}\n"
                output += f"Rows returned: {len(df)}\n\n"
                
                if len(df) == 0:
                    return output + "No data found."
                
                # Format as table
                col_widths = []
                for i, col in enumerate(columns):
                    max_width = len(col)
                    for row in df[:50]:
                        val_len = len(str(row[i]))
                        if val_len > max_width:
                            max_width = val_len
                    col_widths.append(min(max_width, 50))
                
                header = '| '
                separator = '|-'
                for i, col in enumerate(columns):
                    header += col.ljust(col_widths[i]) + ' | '
                    separator += '-' * col_widths[i] + '-|-'
                
                output += header + '\n'
                output += separator + '\n'
                
                for row in df[:100]:  # Limit display to 100 rows
                    row_str = '| '
                    for i, val in enumerate(row):
                        val_str = str(val)[:col_widths[i]]
                        row_str += val_str.ljust(col_widths[i]) + ' | '
                    output += row_str + '\n'
                
                if len(df) > 100:
                    output += f"\n... ({len(df) - 100} more rows)"
                
                return output
        else:
            # For other DML operations, get affected rows
            result.collect()  # Execute the statement
            return f"{statement_type} executed successfully. Use QUERY_HISTORY to check affected rows."
            
    except Exception as e:
        return f"Error executing DML: {str(e)}"
$$;

-- ============================================================================
-- [SECTION_3_QUERY_TOOLS] QUERY ANALYSIS AND HISTORY TOOLS
-- ============================================================================

USE SCHEMA DIGITALSE.QUERY_DEMO;

/*
===============================================================================
QUERY DATA FETCHER - Retrieve Query Operator Statistics
===============================================================================
*/

DROP PROCEDURE IF EXISTS DIGITALSE.QUERY_DEMO.QUERY_DATA_FETCHER(STRING);

CREATE OR REPLACE PROCEDURE QUERY_DATA_FETCHER(
    query_id STRING,
    output_format STRING DEFAULT 'pretty'  -- 'pretty' or 'minified'
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
AS $$
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

def format_bytes(bytes_val: float) -> str:
    """Convert bytes to human-readable format (MB, GB, TB) with consistent formatting."""
    if bytes_val is None or bytes_val == 0:
        return "0.00 MB"
    
    if bytes_val >= 1e12:
        return f"{bytes_val/1e12:.2f} TB"
    elif bytes_val >= 1e9:
        return f"{bytes_val/1e9:.2f} GB"
    else:
        return f"{bytes_val/1e6:.2f} MB"

def truncate_expression(expr: str, max_length: int = 200) -> str:
    """Truncate long expressions to prevent JSON bloat."""
    if not expr or len(expr) <= max_length:
        return expr
    
    # Keep first 80 and last 80 chars for 200 char limit
    keep_chars = (max_length - 20) // 2
    return f"{expr[:keep_chars]}...[truncated]...{expr[-keep_chars:]}"

def parse_parent_operators(parent_ops: Union[str, List, None]) -> Optional[List[int]]:
    """Parse PARENT_OPERATORS field which can be a JSON string, list, or null."""
    if parent_ops is None:
        return None
    
    if isinstance(parent_ops, str):
        try:
            parsed = json.loads(parent_ops)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    
    if isinstance(parent_ops, list):
        return parent_ops
    
    return None

def condense_operator_to_essentials(row, operator_id: int) -> Dict[str, Any]:
    """Extract only essential fields from an operator row."""
    condensed = {
        "operator_id": operator_id,
        "operator_type": row['OPERATOR_TYPE'],
        "parent_operators": parse_parent_operators(row['PARENT_OPERATORS'])
    }
    
    # Add execution time if significant
    if row['EXECUTION_TIME_BREAKDOWN']:
        try:
            exec_breakdown = json.loads(row['EXECUTION_TIME_BREAKDOWN'])
            overall_pct = exec_breakdown.get('overall_percentage', 0)
            if overall_pct > 0:
                condensed['overall_percentage'] = overall_pct
                
                # Find primary time category if execution is significant
                if overall_pct > 5:
                    categories = {
                        'processing': exec_breakdown.get('processing', 0),
                        'sync': exec_breakdown.get('synchronization', 0),
                        'local_io': exec_breakdown.get('local_disk_io', 0),
                        'remote_io': exec_breakdown.get('remote_disk_io', 0),
                        'network': exec_breakdown.get('network_communication', 0),
                        'other': exec_breakdown.get('other', 0)
                    }
                    # Get highest category
                    max_cat = max(categories.items(), key=lambda x: x[1])
                    if max_cat[1] > 0:
                        condensed['primary_time'] = f"{max_cat[0]}:{max_cat[1]}%"
        except json.JSONDecodeError:
            pass
    
    # Add operator statistics
    if row['OPERATOR_STATISTICS']:
        try:
            stats = json.loads(row['OPERATOR_STATISTICS'])
            
            # Row counts
            if 'input_rows' in stats and stats['input_rows']:
                condensed['input_rows'] = stats['input_rows']
            if 'output_rows' in stats and stats['output_rows']:
                condensed['output_rows'] = stats['output_rows']
            
            # I/O metrics
            io = stats.get('io', {})
            if io:
                if io.get('bytes_scanned'):
                    condensed['bytes_scanned'] = format_bytes(io['bytes_scanned'])
                if io.get('percentage_scanned_from_cache'):
                    condensed['cache_hit_rate'] = io['percentage_scanned_from_cache']
                if io.get('bytes_written'):
                    condensed['bytes_written'] = format_bytes(io['bytes_written'])
            
            # Pruning for TableScans
            pruning = stats.get('pruning', {})
            if pruning and pruning.get('partitions_total'):
                scanned = pruning.get('partitions_scanned', 0)
                total = pruning['partitions_total']
                if total > 0:
                    condensed['pruning_efficiency'] = round((1 - scanned/total) * 100, 1)
            
            # Spilling if present
            spilling = stats.get('spilling', {})
            remote = spilling.get('bytes_spilled_remote_storage', 0)
            local = spilling.get('bytes_spilled_local_storage', 0)
            if remote > 0 or local > 0:
                condensed['spilling'] = format_bytes(remote + local)
            
            # DML stats if present
            dml = stats.get('dml', {})
            dml_total = sum([
                dml.get('number_of_rows_inserted', 0),
                dml.get('number_of_rows_updated', 0),
                dml.get('number_of_rows_deleted', 0)
            ])
            if dml_total > 0:
                condensed['dml_rows_affected'] = dml_total
                
        except json.JSONDecodeError:
            pass
    
    # Add key attributes based on operator type
    if row['OPERATOR_ATTRIBUTES']:
        try:
            attrs = json.loads(row['OPERATOR_ATTRIBUTES'])
            
            # TableScan
            if row['OPERATOR_TYPE'] == 'TableScan':
                if 'table_name' in attrs:
                    condensed['table_name'] = attrs['table_name']
                if 'columns' in attrs:
                    condensed['column_count'] = len(attrs['columns'])
                    
            # Joins
            elif row['OPERATOR_TYPE'] in ['InnerJoin', 'LeftOuterJoin', 'RightOuterJoin', 'CartesianJoin']:
                if 'equality_join_condition' in attrs:
                    condensed['join_condition'] = truncate_expression(attrs['equality_join_condition'], 100)
                    
            # Filter
            elif row['OPERATOR_TYPE'] == 'Filter':
                if 'filter_condition' in attrs:
                    condensed['filter_condition'] = truncate_expression(attrs['filter_condition'], 150)
                    
            # Aggregate
            elif row['OPERATOR_TYPE'] in ['Aggregate', 'GroupingSets']:
                functions = attrs.get('functions', [])
                if functions:
                    condensed['aggregate_functions'] = ','.join(functions[:5])
                grouping_keys = attrs.get('grouping_keys', [])
                if grouping_keys:
                    condensed['group_by'] = ','.join(grouping_keys[:3])
                    
            # Sort
            elif row['OPERATOR_TYPE'] in ['Sort', 'SortWithLimit']:
                sort_keys = attrs.get('sort_keys', [])
                if sort_keys:
                    condensed['sort_keys'] = ','.join(sort_keys[:3])
                if row['OPERATOR_TYPE'] == 'SortWithLimit' and 'rows' in attrs:
                    condensed['limit'] = attrs['rows']
                    
            # CreateTableAsSelect
            elif row['OPERATOR_TYPE'] == 'CreateTableAsSelect':
                if 'table_name' in attrs:
                    condensed['target_table'] = attrs['table_name']
                if 'input_expressions' in attrs:
                    expressions = attrs['input_expressions']
                    condensed['expression_count'] = len(expressions)
                    if expressions:
                        # Show first expression sample
                        condensed['sample_expression'] = truncate_expression(expressions[0], 150)
                        
            # DML Operations
            elif row['OPERATOR_TYPE'] in ['Insert', 'Update', 'Delete', 'Merge']:
                if 'table_name' in attrs:
                    condensed['target_table'] = attrs['table_name']
                    
            # Result
            elif row['OPERATOR_TYPE'] == 'Result':
                if 'expressions' in attrs:
                    condensed['output_columns'] = len(attrs['expressions'])
                    
        except json.JSONDecodeError:
            pass
    
    return condensed

def calculate_summary_metrics(operator_stats_list: List[Dict[str, Any]], df) -> Dict[str, Any]:
    """Calculate summary metrics from the raw dataframe."""
    total_bytes_scanned = 0
    total_bytes_written = 0
    total_bytes_spilled = 0
    final_output_rows = 0
    total_dml_rows = 0
    cache_hits = []
    pruning_efficiencies = []
    high_execution_operators = []
    operators_with_spilling = []
    exploding_joins = []
    
    for _, row in df.iterrows():
        operator_id = row['OPERATOR_ID']
        operator_type = row['OPERATOR_TYPE']
        
        # Parse statistics
        if row['OPERATOR_STATISTICS']:
            try:
                stats = json.loads(row['OPERATOR_STATISTICS'])
                
                # Track final output rows from Result operator
                if operator_type == 'Result':
                    input_rows = stats.get('input_rows', 0)
                    if input_rows:
                        final_output_rows = input_rows
                
                # Sum I/O bytes
                io = stats.get('io', {})
                total_bytes_scanned += io.get('bytes_scanned', 0)
                total_bytes_scanned += io.get('external_bytes_scanned', 0)
                total_bytes_written += io.get('bytes_written', 0)
                total_bytes_written += io.get('bytes_written_to_result', 0)
                
                # Cache hits
                cache_hit = io.get('percentage_scanned_from_cache')
                if cache_hit is not None:
                    cache_hits.append(cache_hit)
                
                # Pruning
                pruning = stats.get('pruning', {})
                if pruning and pruning.get('partitions_total'):
                    scanned = pruning.get('partitions_scanned', 0)
                    total = pruning['partitions_total']
                    if total > 0:
                        efficiency = (1 - scanned/total) * 100
                        pruning_efficiencies.append(efficiency)
                
                # Spilling
                spilling = stats.get('spilling', {})
                remote = spilling.get('bytes_spilled_remote_storage', 0)
                local = spilling.get('bytes_spilled_local_storage', 0)
                if remote > 0 or local > 0:
                    total_bytes_spilled += remote + local
                    operators_with_spilling.append({
                        "operator_id": operator_id,
                        "operator_type": operator_type,
                        "bytes_spilled_remote": format_bytes(remote) if remote > 0 else "0 MB",
                        "bytes_spilled_local": format_bytes(local) if local > 0 else "0 MB",
                        "total_bytes_spilled": format_bytes(remote + local)
                    })
                
                # DML rows
                dml = stats.get('dml', {})
                total_dml_rows += dml.get('number_of_rows_inserted', 0)
                total_dml_rows += dml.get('number_of_rows_updated', 0)
                total_dml_rows += dml.get('number_of_rows_deleted', 0)
                
                # Check for exploding joins
                if operator_type in ['Join', 'InnerJoin', 'LeftOuterJoin', 'RightOuterJoin', 'OuterJoin', 'CartesianJoin']:
                    input_rows = stats.get('input_rows', 0)
                    output_rows = stats.get('output_rows', 0)
                    if input_rows > 0 and output_rows > input_rows * 10:
                        exploding_joins.append({
                            "operator_id": operator_id,
                            "operator_type": operator_type,
                            "input_rows": input_rows,
                            "output_rows": output_rows,
                            "multiplication_factor": round(output_rows / input_rows, 2)
                        })
                
            except json.JSONDecodeError:
                pass
        
        # Parse execution time
        if row['EXECUTION_TIME_BREAKDOWN']:
            try:
                exec_breakdown = json.loads(row['EXECUTION_TIME_BREAKDOWN'])
                overall_pct = exec_breakdown.get('overall_percentage', 0)
                if overall_pct > 15:
                    high_execution_operators.append({
                        "id": operator_id,
                        "type": operator_type,
                        "pct": overall_pct
                    })
            except json.JSONDecodeError:
                pass
    
    # Determine query type
    query_type = "SELECT"
    for _, row in df.iterrows():
        if row['OPERATOR_TYPE'] == 'CreateTableAsSelect':
            query_type = "CREATE TABLE AS SELECT"
            break
        elif row['OPERATOR_TYPE'] == 'Insert':
            query_type = "INSERT"
            break
        elif row['OPERATOR_TYPE'] in ['Update', 'Delete', 'Merge']:
            query_type = row['OPERATOR_TYPE'].upper()
            break
    
    # Build summary
    summary = {
        "query_type": query_type,
        "operator_count": len(df),
        "total_bytes_scanned": format_bytes(total_bytes_scanned),
        "total_bytes_written": format_bytes(total_bytes_written),
        "final_output_rows": final_output_rows
    }
    
    # Add DML metrics if relevant
    if total_dml_rows > 0:
        summary["dml_rows_affected"] = {
            "total": total_dml_rows
        }
    else:
        summary["dml_rows_affected"] = {
            "inserted": 0,
            "updated": 0,
            "deleted": 0
        }
    
    # Add spilling information
    if total_bytes_spilled > 0:
        summary["total_bytes_spilled"] = format_bytes(total_bytes_spilled)
        summary["spilling_operator_count"] = len(operators_with_spilling)
    
    # Always include cache and pruning statistics
    summary["average_cache_hit_rate"] = round(sum(cache_hits) / len(cache_hits), 1) if cache_hits else 0.0
    summary["average_pruning_efficiency"] = round(sum(pruning_efficiencies) / len(pruning_efficiencies), 1) if pruning_efficiencies else 0.0
    
    # Convert performance issues to JSON strings like operators
    high_execution_json = [json.dumps(op, separators=(',', ':')) for op in high_execution_operators]
    exploding_joins_json = [json.dumps(op, separators=(',', ':')) for op in exploding_joins]
    spilling_operators_json = [json.dumps(op, separators=(',', ':')) for op in operators_with_spilling]
    
    # Add detailed performance issues section with JSON strings
    performance_issues = {
        "high_execution_time_operators_count": len(high_execution_operators),
        "high_execution_time_operators_json": high_execution_json,
        "exploding_joins_count": len(exploding_joins),
        "exploding_joins_json": exploding_joins_json,
        "operators_with_spilling_count": len(operators_with_spilling),
        "operators_with_spilling_json": spilling_operators_json,
        "external_function_operators_count": 0,
        "external_function_operators_json": [],
        "total_operators": len(df)
    }
    summary["performance_issues"] = performance_issues
    
    # Add external function summary
    summary["external_function_summary"] = {
        "total_calls": 0,
        "total_errors": 0,
        "success_rate_percentage": None,
        "operators_using_external_functions": 0
    }
    
    return summary

def main(session, query_id: str, output_format: str = 'pretty') -> str:
    """Main handler function for fetching query data."""
    try:
        # Validate query ID format
        try:
            uuid.UUID(query_id)
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": "Invalid Query ID format. Please provide a valid UUID.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Get operator statistics
        query = f"SELECT * FROM TABLE(GET_QUERY_OPERATOR_STATS('{query_id}'))"
        df = session.sql(query).to_pandas()
        
        if df.empty:
            return json.dumps({
                "status": "error",
                "error": "No operator statistics found for the provided Query ID.",
                "query_id": query_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Calculate summary metrics from raw data
        summary_metrics = calculate_summary_metrics(None, df)
        
        # Process each operator into a condensed JSON string
        operator_json_strings = []
        for _, row in df.iterrows():
            operator_id = row['OPERATOR_ID']
            condensed_op = condense_operator_to_essentials(row, operator_id)
            
            # Convert to JSON string (minified)
            op_json_str = json.dumps(condensed_op, separators=(',', ':'))
            operator_json_strings.append(op_json_str)
        
        # Build final response with operators as JSON strings
        response = {
            "status": "success",
            "query_id": query_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_metrics": summary_metrics,
            "operators_json_lines": operator_json_strings,
            "parse_instructions": "Each element in operators_json_lines is a JSON string. Parse with json.loads() to get operator dict."
        }
        
        # Return based on format preference
        if output_format == 'minified':
            return json.dumps(response, separators=(',', ':'))
        else:
            return json.dumps(response, indent=2)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        return json.dumps({
            "status": "error",
            "error": str(e),
            "details": error_details.replace('\n', ' | '),
            "query_id": query_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
$$;

/*
===============================================================================
QUERY HISTORY FETCHER - Retrieve Query Text and Metadata
===============================================================================
*/

USE SCHEMA DIGITALSE.TOOLS;

DROP FUNCTION IF EXISTS QUERY_TEXT(VARCHAR);

CREATE OR REPLACE PROCEDURE QUERY_TEXT(query_id VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'get_query_text'
COMMENT = 'Retrieves query execution details from Snowflake query history for a specific query_id'
AS
$$
import json

def get_query_text(session, query_id):
    """Retrieves query execution details for a specific query_id."""
    try:
        # Validate input
        if not query_id or not query_id.strip():
            return json.dumps({
                "status": "error",
                "message": "Query ID cannot be empty"
            })
        
        # Clean the query_id
        query_id = query_id.strip().strip("'\"")
        
        # Query the ACCOUNT_USAGE.QUERY_HISTORY view
        query = f"""
        SELECT 
            QUERY_ID,
            QUERY_TEXT,
            USER_NAME,
            ROLE_NAME,
            WAREHOUSE_SIZE,
            QUERY_LOAD_PERCENT
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE QUERY_ID = '{query_id}'
          AND QUERY_ID IS NOT NULL
        ORDER BY START_TIME DESC
        LIMIT 1
        """
        
        # Execute the query
        result = session.sql(query).collect()
        
        # Check if we found the query
        if not result:
            return json.dumps({
                "status": "error",
                "message": f"Query ID '{query_id}' not found in query history. Note: ACCOUNT_USAGE may have up to 45 minutes latency."
            })
        
        # Extract the row
        row = result[0]
        
        # Get the query text and format it
        query_text = str(row['QUERY_TEXT']) if row['QUERY_TEXT'] else None
        
        # Create a more readable version
        if query_text:
            import re
            # Remove excessive whitespace
            formatted_text = re.sub(r'\n\s+', ' ', query_text)
            formatted_text = re.sub(r'\s+', ' ', formatted_text)
            formatted_text = formatted_text.strip()
            query_text_formatted = formatted_text
        else:
            query_text_formatted = None
        
        # Build the response
        response = {
            "status": "success",
            "query_id": str(row['QUERY_ID']),
            "query_text_formatted": query_text_formatted,
            "user_name": str(row['USER_NAME']) if row['USER_NAME'] else None,
            "role_name": str(row['ROLE_NAME']) if row['ROLE_NAME'] else None,
            "warehouse_size": str(row['WAREHOUSE_SIZE']) if row['WAREHOUSE_SIZE'] else None,
            "query_load_percent": float(row['QUERY_LOAD_PERCENT']) if row['QUERY_LOAD_PERCENT'] is not None else None
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error retrieving query: {str(e)}"
        })
$$;

-- ============================================================================
-- [SECTION_4_PROMPTING_TOOLS] AI PROMPTING AND OPTIMIZATION TOOLS
-- ============================================================================

USE SCHEMA DIGITALSE.QUERY_DEMO;

/*
===============================================================================
FIELD DEFINITIONS - Query Operator Field Definitions
===============================================================================
*/

CREATE OR REPLACE PROCEDURE FIELD_DEFINITIONS()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
AS $$
import json

def get_field_definitions() -> str:
    """Return comprehensive field definitions for Snowflake Intelligence."""
    return """
    QUERY OPERATOR STATISTICS FIELD DEFINITIONS:

    CORE FIELDS:
    - QUERY_ID: VARCHAR, The query ID, an internal, system-generated identifier for the SQL statement.
    - STEP_ID: NUMBER(38, 0), Identifier of the step in the query plan.
    - OPERATOR_ID: NUMBER(38, 0), The operator's identifier, unique within the query, starting at 0.
    - PARENT_OPERATORS: ARRAY of NUMBER(38, 0), Identifiers of the parent operators, or NULL for the final operator (usually Result).
    - OPERATOR_TYPE: VARCHAR, The type of query operator including TableScan, Join, Filter, Sort, Aggregate, etc.

    OPERATOR_STATISTICS (JSON object with nested structures):

    I/O Statistics:
    - input_rows: INTEGER, Number of input rows processed by the operator.
    - output_rows: INTEGER, Number of output rows produced by the operator.
    - scan_progress: DOUBLE, Percentage of table scanned (0.0-1.0).
    - io.bytes_scanned: INTEGER, Number of bytes scanned.
    - io.percentage_scanned_from_cache: DOUBLE, Percentage of data scanned from cache (0-100).
    - io.bytes_written: INTEGER, Number of bytes written.

    Pruning Statistics:
    - pruning.partitions_scanned: INTEGER, Number of partitions scanned.
    - pruning.partitions_total: INTEGER, Total number of partitions available.

    EXECUTION_TIME_BREAKDOWN (JSON object):
    - overall_percentage: DOUBLE, Percentage of total query execution time consumed by this operator.
    - processing: DOUBLE, Time spent processing the data by the CPU.
    - local_disk_io: DOUBLE, Time waiting for local disk access.
    - remote_disk_io: DOUBLE, Time waiting for remote disk access.
    - network_communication: DOUBLE, Time waiting for network data transfer.

    For complete field definitions, refer to Snowflake documentation.
    """

def main(session) -> str:
    """Main handler function that returns field definitions."""
    return json.dumps({
        "status": "success",
        "field_definitions": get_field_definitions()
    }, indent=2)
$$;

/*
===============================================================================
ANALYSIS GUIDANCE - Performance Analysis Guidelines
===============================================================================
*/

CREATE OR REPLACE PROCEDURE ANALYSIS_GUIDANCE()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
AS $$
import json

def get_analysis_guidance() -> str:
    """Return guidance for Snowflake Intelligence on how to analyze the data."""
    return """
    PERFORMANCE THRESHOLDS:

    EXECUTION TIME:
    • GREAT: Operators consuming <5% of total execution time
    • GOOD: Operators consuming 5-15% of total execution time
    • POOR: Operators consuming 15-30% of total execution time
    • CRITICAL: Operators consuming >30% of total execution time

    TABLE SCAN EFFICIENCY:
    • GREAT: Pruning efficiency >80% AND cache hit rate >90%
    • GOOD:  Pruning efficiency 60-80% OR cache hit rate 70-90%
    • POOR:  Pruning efficiency 30-60% OR cache hit rate 40-70%
    • CRITICAL: Pruning efficiency <30% OR cache hit rate <40%

    JOIN PERFORMANCE:
    • GREAT: Row multiplication factor <1.2 (minimal row expansion)
    • GOOD: Row multiplication factor 1.2-1.5
    • POOR: Row multiplication factor 1.5-2.0
    • CRITICAL: Row multiplication factor >2.0 (join explosion)

    SPILLING:
    • GREAT: No spilling detected
    • GOOD: Local spilling only (<100 MB)
    • POOR: Local spilling (100 MB - 1 GB)
    • CRITICAL: Remote spilling OR >1 GB spilled
    """

def main(session) -> str:
    """Main handler function that returns analysis guidance."""
    return json.dumps({
        "status": "success",
        "analysis_guidance": get_analysis_guidance()
    }, indent=2)
$$;

/*
===============================================================================
QUERY OPTIMIZE WORKFLOW - Query Optimization Guidance
===============================================================================
*/

CREATE OR REPLACE PROCEDURE QUERY_OPTIMIZE_WORKFLOW()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
AS $$
import json

def get_optimization_workflow() -> dict:
    """Return controlled query optimization workflow for Snowflake Intelligence."""
    return {
        "prompt_id": "query_optimization_simple_v1",
        "prompt_type": "query_optimization",
        "metadata": {
            "version": "1.0.0",
            "created": "2025-01-18",
            "workflow_name": "Simple Query Optimizer",
            "focus": "Quick optimization without extensive metadata gathering"
        },
        "star_structure": {
            "situation": {
                "context": "You need to optimize a Snowflake SQL query efficiently",
                "approach": "Analyze query structure with limited metadata gathering",
                "critical_requirements": {
                    "platform": "⚠️ SNOWFLAKE SQL ONLY ⚠️",
                    "results": "Must return identical results"
                }
            }
        }
    }

def main(session) -> str:
    """Main handler function that returns controlled query optimization workflow."""
    workflow = get_optimization_workflow()
    return json.dumps(workflow, indent=2)
$$;

-- ============================================================================
-- [SECTION_5_SEMANTIC_VIEWS] ACCOUNT USAGE SEMANTIC VIEWS
-- ============================================================================

USE SCHEMA DIGITALSE.PUBLIC;

-- First, create views that reference the actual SNOWFLAKE.ACCOUNT_USAGE tables
-- These views allow the semantic view to reference the data

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.COLUMN_QUERY_PRUNING_HISTORY AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.COLUMN_QUERY_PRUNING_HISTORY;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.QUERY_ACCELERATION_ELIGIBLE AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_ACCELERATION_ELIGIBLE;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.QUERY_ATTRIBUTION_HISTORY AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.QUERY_HISTORY AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.QUERY_INSIGHTS AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_INSIGHTS;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.TABLE_PRUNING_HISTORY AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_PRUNING_HISTORY;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.TABLE_QUERY_PRUNING_HISTORY AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_QUERY_PRUNING_HISTORY;

CREATE OR REPLACE VIEW DIGITALSE.PUBLIC.WAREHOUSE_LOAD_HISTORY AS
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_LOAD_HISTORY;

-- Now create the semantic view that references these views
create or replace semantic view DIGITALSE.PUBLIC.ACCOUNT_USAGE_SEMANTIC_VIEW
	tables (
		COLUMN_QUERY_PRUNING_HISTORY comment='Use this Account Usage view to gain a better understanding of data access patterns during query execution, including some column-level details, such as the “access type” and candidate search optimization expressions that are potentially beneficial.

You can use this view in combination with the TABLE_QUERY_PRUNING_HISTORY view. For example, you can identify access to target tables by using the TABLE_QUERY_PRUNING_HISTORY view, then identify frequently used columns on those tables by using the COLUMN_QUERY_PRUNING_HISTORY view.

Each row in this view represents the query pruning history for a specific column within a given time interval. The data is aggregated per column, per table, per interval, and includes metrics such as the number of queries executed, partitions scanned, partitions pruned, rows scanned, rows pruned, and rows matched.',
		QUERY_ACCELERATION_ELIGIBLE comment='This Account Usage view can be used to identify queries that are eligible for the query acceleration service (QAS).',
		QUERY_ATTRIBUTION_HISTORY primary key (QUERY_ID) comment='This Account Usage view can be used to determine the compute cost of a given query run on warehouses in your account in the last 365 days (1 year).',
		QUERY_HISTORY primary key (QUERY_ID) comment='This Account Usage view can be used to query Snowflake query history by various dimensions (time range, session, user, warehouse, and so on) within the last 365 days (1 year).',
		QUERY_INSIGHTS primary key (INSIGHT_INSTANCE_ID) unique (QUERY_ID) comment='This Account Usage view displays a row for each insight produced for a query.',
		TABLE_PRUNING_HISTORY comment='This Account Usage view can be used to determine the efficiency of pruning for all tables, and to understand how a table’s default (natural) ordering of data affects pruning.

You can compare the number of partitions pruned (PARTITIONS_PRUNED) to the total number of partitions scanned and pruned (PARTITIONS_SCANNED + PARTITIONS_PRUNED).

Each row in this view represents the pruning history for a specific table within a given time interval. The data is aggregated by time interval and includes information about the number of scans, partitions scanned, partitions pruned, rows scanned, and rows pruned.

You can also use this view to compare the effects on pruning before and after enabling Automatic Clustering and search optimization for a table.',
		TABLE_QUERY_PRUNING_HISTORY comment='Use this Account Usage view to gain a better understanding of data access patterns during query execution.

You can use this view in combination with the COLUMN_QUERY_PRUNING_HISTORY view. For example, you can identify access to target tables by using the TABLE_QUERY_PRUNING_HISTORY view, then identify frequently used columns on those tables by using the COLUMN_QUERY_PRUNING_HISTORY view.

In particular, these views can help you make a more educated choice for clustering keys.

Each row in this view represents the query pruning history for a specific table within a given time interval. The data is aggregated by time interval and includes information about the number of queries executed, partitions scanned, partitions pruned, rows scanned, rows pruned, and rows matched.',
		WAREHOUSE_LOAD_HISTORY comment='This Account Usage view can be used to analyze the workload on your warehouse within a specified date range.'
	)
	relationships (
		QUERY_HISTORY_TO_QUERY_INSIGHTS as QUERY_HISTORY(QUERY_ID) references QUERY_INSIGHTS(QUERY_ID),
		QUERY_HISTORY_TO_QUERY_ATTRIBUTION_HISTORY as QUERY_HISTORY(QUERY_ID) references QUERY_ATTRIBUTION_HISTORY(QUERY_ID)
	)
	facts (
		COLUMN_QUERY_PRUNING_HISTORY.AGGREGATE_QUERY_COMPILATION_TIME as AGGREGATE_QUERY_COMPILATION_TIME comment='Total compilation time for queries defined by NUM_QUERIES.',
		COLUMN_QUERY_PRUNING_HISTORY.AGGREGATE_QUERY_ELAPSED_TIME as AGGREGATE_QUERY_ELAPSED_TIME comment='Total elapsed time for queries defined by NUM_QUERIES.',
		COLUMN_QUERY_PRUNING_HISTORY.AGGREGATE_QUERY_EXECUTION_TIME as AGGREGATE_QUERY_EXECUTION_TIME comment='Total execution time for queries defined by NUM_QUERIES.',
		COLUMN_QUERY_PRUNING_HISTORY.COLUMN_ID as COLUMN_ID comment='Internal/system-generated identifier for the column.',
		COLUMN_QUERY_PRUNING_HISTORY.DATABASE_ID as DATABASE_ID comment='Internal/system-generated identifier for the database.',
		COLUMN_QUERY_PRUNING_HISTORY.NUM_QUERIES as NUM_QUERIES comment='Number of queries that scanned this column in the given time interval.',
		COLUMN_QUERY_PRUNING_HISTORY.PARTITIONS_PRUNED as PARTITIONS_PRUNED comment='Number of partitions pruned on this table for queries defined by NUM_QUERIES. These partitions were eliminated during query processing and not scanned, improving the efficiency of the query.',
		COLUMN_QUERY_PRUNING_HISTORY.PARTITIONS_SCANNED as PARTITIONS_SCANNED comment='Number of partitions scanned on this table for queries defined by NUM_QUERIES.',
		COLUMN_QUERY_PRUNING_HISTORY.ROWS_MATCHED as ROWS_MATCHED comment='Number of rows that matched the WHERE clause filters while scanning this table for the queries defined by NUM_QUERIES.',
		COLUMN_QUERY_PRUNING_HISTORY.ROWS_PRUNED as ROWS_PRUNED comment='Number of rows pruned on this table for queries defined by NUM_QUERIES. These rows were eliminated during query processing and not scanned, improving the efficiency of the query.',
		COLUMN_QUERY_PRUNING_HISTORY.ROWS_SCANNED as ROWS_SCANNED comment='Number of rows scanned on this table for queries defined by NUM_QUERIES.',
		COLUMN_QUERY_PRUNING_HISTORY.SCHEMA_ID as SCHEMA_ID comment='Internal/system-generated identifier for the schema.',
		COLUMN_QUERY_PRUNING_HISTORY.TABLE_ID as TABLE_ID comment='Internal/system-generated identifier for the table.',
		COLUMN_QUERY_PRUNING_HISTORY.WAREHOUSE_ID as WAREHOUSE_ID comment='Internal/system-generated identifier for the warehouse used to run the query.',
		QUERY_ACCELERATION_ELIGIBLE.ELIGIBLE_QUERY_ACCELERATION_TIME as ELIGIBLE_QUERY_ACCELERATION_TIME comment='Amount of time the query was eligible for query acceleration (in milliseconds).',
		QUERY_ACCELERATION_ELIGIBLE.QUERY_HASH_VERSION as QUERY_HASH_VERSION comment='Version number of the query hash algorithm.',
		QUERY_ACCELERATION_ELIGIBLE.QUERY_PARAMETERIZED_HASH_VERSION as QUERY_PARAMETERIZED_HASH_VERSION comment='Version number of the parameterized query hash algorithm.',
		QUERY_ACCELERATION_ELIGIBLE.UPPER_LIMIT_SCALE_FACTOR as UPPER_LIMIT_SCALE_FACTOR comment='Upper limit scale factor for query acceleration eligibility.',
		QUERY_ATTRIBUTION_HISTORY.CREDITS_ATTRIBUTED_COMPUTE as CREDITS_ATTRIBUTED_COMPUTE comment='Number of credits billed for warehouse compute resources used by the query.',
		QUERY_ATTRIBUTION_HISTORY.CREDITS_USED_QUERY_ACCELERATION as CREDITS_USED_QUERY_ACCELERATION comment='Number of credits billed for the query acceleration service used by the query.',
		QUERY_ATTRIBUTION_HISTORY.WAREHOUSE_ID as WAREHOUSE_ID comment='Internal/system-generated identifier for the warehouse used to run the query.',
		QUERY_HISTORY.BYTES_DELETED as BYTES_DELETED comment='Number of bytes deleted by the query.',
		QUERY_HISTORY.BYTES_READ_FROM_RESULT as BYTES_READ_FROM_RESULT comment='Number of bytes read from a result object.',
		QUERY_HISTORY.BYTES_SCANNED as BYTES_SCANNED comment='Number of bytes scanned by the query.',
		QUERY_HISTORY.BYTES_SENT_OVER_THE_NETWORK as BYTES_SENT_OVER_THE_NETWORK comment='Number of bytes sent over the network for the query.',
		QUERY_HISTORY.BYTES_SPILLED_TO_LOCAL_STORAGE as BYTES_SPILLED_TO_LOCAL_STORAGE comment='Number of bytes spilled to local storage during query processing.',
		QUERY_HISTORY.BYTES_SPILLED_TO_REMOTE_STORAGE as BYTES_SPILLED_TO_REMOTE_STORAGE comment='Number of bytes spilled to remote storage during query processing.',
		QUERY_HISTORY.BYTES_WRITTEN as BYTES_WRITTEN comment='Number of bytes written by the query.',
		QUERY_HISTORY.BYTES_WRITTEN_TO_RESULT as BYTES_WRITTEN_TO_RESULT comment='Number of bytes written to a result object.',
		QUERY_HISTORY.CHILD_QUERIES_WAIT_TIME as CHILD_QUERIES_WAIT_TIME comment='Time spent waiting for child queries to complete (in milliseconds).',
		QUERY_HISTORY.CLUSTER_NUMBER as CLUSTER_NUMBER comment='Cluster number within the multi-cluster warehouse that processed the query.',
		QUERY_HISTORY.COMPILATION_TIME as COMPILATION_TIME comment='Query compilation time in milliseconds.',
		QUERY_HISTORY.CREDITS_USED_CLOUD_SERVICES as CREDITS_USED_CLOUD_SERVICES comment='Number of credits used for cloud services.',
		QUERY_HISTORY.DATABASE_ID as DATABASE_ID comment='Internal/system-generated identifier for the database.',
		QUERY_HISTORY.EXECUTION_TIME as EXECUTION_TIME comment='Query execution time in milliseconds.',
		QUERY_HISTORY.EXTERNAL_FUNCTION_TOTAL_INVOCATIONS as EXTERNAL_FUNCTION_TOTAL_INVOCATIONS comment='Total number of external function invocations in the query.',
		QUERY_HISTORY.EXTERNAL_FUNCTION_TOTAL_RECEIVED_BYTES as EXTERNAL_FUNCTION_TOTAL_RECEIVED_BYTES comment='Total number of bytes received from external functions.',
		QUERY_HISTORY.EXTERNAL_FUNCTION_TOTAL_RECEIVED_ROWS as EXTERNAL_FUNCTION_TOTAL_RECEIVED_ROWS comment='Total number of rows received from external functions.',
		QUERY_HISTORY.EXTERNAL_FUNCTION_TOTAL_SENT_BYTES as EXTERNAL_FUNCTION_TOTAL_SENT_BYTES comment='Total number of bytes sent to external functions.',
		QUERY_HISTORY.EXTERNAL_FUNCTION_TOTAL_SENT_ROWS as EXTERNAL_FUNCTION_TOTAL_SENT_ROWS comment='Total number of rows sent to external functions.',
		QUERY_HISTORY.FAULT_HANDLING_TIME as FAULT_HANDLING_TIME,
		QUERY_HISTORY.INBOUND_DATA_TRANSFER_BYTES as INBOUND_DATA_TRANSFER_BYTES,
		QUERY_HISTORY.LIST_EXTERNAL_FILES_TIME as LIST_EXTERNAL_FILES_TIME,
		QUERY_HISTORY.OUTBOUND_DATA_TRANSFER_BYTES as OUTBOUND_DATA_TRANSFER_BYTES,
		QUERY_HISTORY.PARTITIONS_SCANNED as PARTITIONS_SCANNED comment='Number of partitions scanned by the query.',
		QUERY_HISTORY.PARTITIONS_TOTAL as PARTITIONS_TOTAL comment='Total number of partitions in the table(s) accessed by the query.',
		QUERY_HISTORY.PERCENTAGE_SCANNED_FROM_CACHE as PERCENTAGE_SCANNED_FROM_CACHE comment='Percentage of data scanned from cache vs. storage.',
		QUERY_HISTORY.QUERY_ACCELERATION_BYTES_SCANNED as QUERY_ACCELERATION_BYTES_SCANNED,
		QUERY_HISTORY.QUERY_ACCELERATION_PARTITIONS_SCANNED as QUERY_ACCELERATION_PARTITIONS_SCANNED,
		QUERY_HISTORY.QUERY_ACCELERATION_UPPER_LIMIT_SCALE_FACTOR as QUERY_ACCELERATION_UPPER_LIMIT_SCALE_FACTOR,
		QUERY_HISTORY.QUERY_HASH_VERSION as QUERY_HASH_VERSION,
		QUERY_HISTORY.QUERY_LOAD_PERCENT as QUERY_LOAD_PERCENT,
		QUERY_HISTORY.QUERY_PARAMETERIZED_HASH_VERSION as QUERY_PARAMETERIZED_HASH_VERSION,
		QUERY_HISTORY.QUERY_RETRY_TIME as QUERY_RETRY_TIME,
		QUERY_HISTORY.QUEUED_OVERLOAD_TIME as QUEUED_OVERLOAD_TIME,
		QUERY_HISTORY.QUEUED_PROVISIONING_TIME as QUEUED_PROVISIONING_TIME,
		QUERY_HISTORY.QUEUED_REPAIR_TIME as QUEUED_REPAIR_TIME,
		QUERY_HISTORY.ROWS_DELETED as ROWS_DELETED,
		QUERY_HISTORY.ROWS_INSERTED as ROWS_INSERTED,
		QUERY_HISTORY.ROWS_PRODUCED as ROWS_PRODUCED,
		QUERY_HISTORY.ROWS_UNLOADED as ROWS_UNLOADED,
		QUERY_HISTORY.ROWS_UPDATED as ROWS_UPDATED,
		QUERY_HISTORY.ROWS_WRITTEN_TO_RESULT as ROWS_WRITTEN_TO_RESULT,
		QUERY_HISTORY.SCHEMA_ID as SCHEMA_ID,
		QUERY_HISTORY.SESSION_ID as SESSION_ID,
		QUERY_HISTORY.TOTAL_ELAPSED_TIME as TOTAL_ELAPSED_TIME comment='Total elapsed time for the query (in milliseconds).',
		QUERY_HISTORY.TRANSACTION_BLOCKED_TIME as TRANSACTION_BLOCKED_TIME,
		QUERY_HISTORY.TRANSACTION_ID as TRANSACTION_ID,
		QUERY_HISTORY.USER_DATABASE_ID as USER_DATABASE_ID,
		QUERY_HISTORY.USER_SCHEMA_ID as USER_SCHEMA_ID,
		QUERY_HISTORY.WAREHOUSE_ID as WAREHOUSE_ID,
		QUERY_INSIGHTS.TOTAL_ELAPSED_TIME as TOTAL_ELAPSED_TIME comment='Total elapsed time for the query that generated this insight (in milliseconds).',
		QUERY_INSIGHTS.WAREHOUSE_ID as WAREHOUSE_ID comment='Internal/system-generated identifier for the warehouse.',
		TABLE_PRUNING_HISTORY.DATABASE_ID as DATABASE_ID comment='Internal/system-generated identifier for the database.',
		TABLE_PRUNING_HISTORY.NUM_SCANS as NUM_SCANS comment='Number of table scans performed during the time interval.',
		TABLE_PRUNING_HISTORY.PARTITIONS_PRUNED as PARTITIONS_PRUNED comment='Number of partitions that were pruned (eliminated from scanning) during the time interval.',
		TABLE_PRUNING_HISTORY.PARTITIONS_SCANNED as PARTITIONS_SCANNED comment='Number of partitions that were scanned during the time interval.',
		TABLE_PRUNING_HISTORY.ROWS_PRUNED as ROWS_PRUNED comment='Number of rows that were pruned (eliminated from scanning) during the time interval.',
		TABLE_PRUNING_HISTORY.ROWS_SCANNED as ROWS_SCANNED comment='Number of rows that were scanned during the time interval.',
		TABLE_PRUNING_HISTORY.SCHEMA_ID as SCHEMA_ID comment='Internal/system-generated identifier for the schema.',
		TABLE_PRUNING_HISTORY.TABLE_ID as TABLE_ID comment='Internal/system-generated identifier for the table.',
		TABLE_QUERY_PRUNING_HISTORY.AGGREGATE_QUERY_COMPILATION_TIME as AGGREGATE_QUERY_COMPILATION_TIME comment='Total compilation time for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.AGGREGATE_QUERY_ELAPSED_TIME as AGGREGATE_QUERY_ELAPSED_TIME comment='Total elapsed time for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.AGGREGATE_QUERY_EXECUTION_TIME as AGGREGATE_QUERY_EXECUTION_TIME comment='Total execution time for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.DATABASE_ID as DATABASE_ID comment='Internal/system-generated identifier for the database.',
		TABLE_QUERY_PRUNING_HISTORY.NUM_QUERIES as NUM_QUERIES comment='Number of queries that scanned this table in the given time interval.',
		TABLE_QUERY_PRUNING_HISTORY.PARTITIONS_PRUNED as PARTITIONS_PRUNED comment='Number of partitions pruned on this table for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.PARTITIONS_SCANNED as PARTITIONS_SCANNED comment='Number of partitions scanned on this table for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.ROWS_MATCHED as ROWS_MATCHED comment='Number of rows that matched the WHERE clause filters while scanning this table for the queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.ROWS_PRUNED as ROWS_PRUNED comment='Number of rows pruned on this table for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.ROWS_SCANNED as ROWS_SCANNED comment='Number of rows scanned on this table for queries defined by NUM_QUERIES.',
		TABLE_QUERY_PRUNING_HISTORY.SCHEMA_ID as SCHEMA_ID comment='Internal/system-generated identifier for the schema.',
		TABLE_QUERY_PRUNING_HISTORY.TABLE_ID as TABLE_ID comment='Internal/system-generated identifier for the table.',
		TABLE_QUERY_PRUNING_HISTORY.WAREHOUSE_ID as WAREHOUSE_ID comment='Internal/system-generated identifier for the warehouse used to run the query.',
		WAREHOUSE_LOAD_HISTORY.AVG_BLOCKED as AVG_BLOCKED comment='Average number of queries blocked by a transaction lock during the specified time interval.',
		WAREHOUSE_LOAD_HISTORY.AVG_QUEUED_LOAD as AVG_QUEUED_LOAD comment='Average number of queries queued due to the warehouse being overloaded during the specified time interval.',
		WAREHOUSE_LOAD_HISTORY.AVG_QUEUED_PROVISIONING as AVG_QUEUED_PROVISIONING comment='Average number of queries queued because the warehouse was being provisioned during the specified time interval.',
		WAREHOUSE_LOAD_HISTORY.AVG_RUNNING as AVG_RUNNING comment='Average number of queries running concurrently during the specified time interval.',
		WAREHOUSE_LOAD_HISTORY.WAREHOUSE_ID as WAREHOUSE_ID comment='Internal/system-generated identifier for the warehouse.'
	)
	dimensions (
		COLUMN_QUERY_PRUNING_HISTORY.ACCESS_TYPE as ACCESS_TYPE comment='Specifies whether the column is used in a filter condition (WHERE) or join condition (JOIN).',
		COLUMN_QUERY_PRUNING_HISTORY.COLUMN_NAME as COLUMN_NAME comment='Name of the column.',
		COLUMN_QUERY_PRUNING_HISTORY.DATABASE_NAME as DATABASE_NAME comment='Database that the schema belongs to.',
		COLUMN_QUERY_PRUNING_HISTORY.INTERVAL_END_TIME as INTERVAL_END_TIME comment='End time for the time interval.',
		COLUMN_QUERY_PRUNING_HISTORY.INTERVAL_START_TIME as INTERVAL_START_TIME comment='Start time for the time interval.',
		COLUMN_QUERY_PRUNING_HISTORY.QUERY_HASH as QUERY_HASH comment='Hash value computed based on the canonicalized SQL text of the query.',
		COLUMN_QUERY_PRUNING_HISTORY.QUERY_PARAMETERIZED_HASH as QUERY_PARAMETERIZED_HASH comment='Hash value computed based on the parameterized query.',
		COLUMN_QUERY_PRUNING_HISTORY.SCHEMA_NAME as SCHEMA_NAME comment='Schema that the table belongs to.',
		COLUMN_QUERY_PRUNING_HISTORY.SEARCH_OPTIMIZATION_SUPPORTED_EXPRESSIONS as SEARCH_OPTIMIZATION_SUPPORTED_EXPRESSIONS comment='List of supported search optimization expressions on this column that could potentially speed up scanning this table.',
		COLUMN_QUERY_PRUNING_HISTORY.TABLE_NAME as TABLE_NAME comment='Name of the table.',
		COLUMN_QUERY_PRUNING_HISTORY.VARIANT_PATH as VARIANT_PATH comment='Path to a field in a VARIANT, OBJECT, or ARRAY column.',
		COLUMN_QUERY_PRUNING_HISTORY.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the warehouse used to run the query.',
		QUERY_ACCELERATION_ELIGIBLE.END_TIME as END_TIME comment='End time of the query execution.',
		QUERY_ACCELERATION_ELIGIBLE.QUERY_HASH as QUERY_HASH comment='The hash value computed based on the canonicalized SQL text.',
		QUERY_ACCELERATION_ELIGIBLE.QUERY_ID as QUERY_ID comment='Unique identifier for the query.',
		QUERY_ACCELERATION_ELIGIBLE.QUERY_PARAMETERIZED_HASH as QUERY_PARAMETERIZED_HASH comment='Hash value computed based on the parameterized query.',
		QUERY_ACCELERATION_ELIGIBLE.QUERY_TEXT as QUERY_TEXT comment='SQL text of the query.',
		QUERY_ACCELERATION_ELIGIBLE.START_TIME as START_TIME comment='Start time of the query execution.',
		QUERY_ACCELERATION_ELIGIBLE.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the warehouse used to run the query.',
		QUERY_ACCELERATION_ELIGIBLE.WAREHOUSE_SIZE as WAREHOUSE_SIZE comment='Size of the warehouse used to run the query.',
		QUERY_ATTRIBUTION_HISTORY.END_TIME as END_TIME comment='End time of the query execution.',
		QUERY_ATTRIBUTION_HISTORY.PARENT_QUERY_ID as PARENT_QUERY_ID comment='Query ID of the parent query, if this query was executed as part of a child query.',
		QUERY_ATTRIBUTION_HISTORY.QUERY_HASH as QUERY_HASH comment='Hash value computed based on the canonicalized SQL text of the query.',
		QUERY_ATTRIBUTION_HISTORY.QUERY_ID as QUERY_ID comment='Unique identifier for the query.',
		QUERY_ATTRIBUTION_HISTORY.QUERY_PARAMETERIZED_HASH as QUERY_PARAMETERIZED_HASH comment='Hash value computed based on the parameterized query.',
		QUERY_ATTRIBUTION_HISTORY.QUERY_TAG as QUERY_TAG comment='Query tag set for the query.',
		QUERY_ATTRIBUTION_HISTORY.ROOT_QUERY_ID as ROOT_QUERY_ID comment='Query ID of the root query in a multi-statement transaction.',
		QUERY_ATTRIBUTION_HISTORY.START_TIME as START_TIME comment='Start time of the query execution.',
		QUERY_ATTRIBUTION_HISTORY.USER_NAME as USER_NAME comment='User who submitted the query.',
		QUERY_ATTRIBUTION_HISTORY.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the warehouse used to run the query.',
		QUERY_HISTORY.DATABASE_NAME as DATABASE_NAME comment='Database in which the SQL statement executed.',
		QUERY_HISTORY.END_TIME as END_TIME comment='Time when the query finished executing.',
		QUERY_HISTORY.ERROR_CODE as ERROR_CODE comment='Error code, if the query returned an error.',
		QUERY_HISTORY.ERROR_MESSAGE as ERROR_MESSAGE comment='Error message, if the query returned an error.',
		QUERY_HISTORY.EXECUTION_STATUS as EXECUTION_STATUS comment='Execution status of the query (SUCCESS, FAIL, etc.).',
		QUERY_HISTORY.INBOUND_DATA_TRANSFER_CLOUD as INBOUND_DATA_TRANSFER_CLOUD comment='Cloud provider for inbound data transfer.',
		QUERY_HISTORY.INBOUND_DATA_TRANSFER_REGION as INBOUND_DATA_TRANSFER_REGION comment='Source region for inbound data transfer.',
		QUERY_HISTORY.IS_CLIENT_GENERATED_STATEMENT as IS_CLIENT_GENERATED_STATEMENT comment='Indicates whether the query statement was generated by the client.',
		QUERY_HISTORY.OUTBOUND_DATA_TRANSFER_CLOUD as OUTBOUND_DATA_TRANSFER_CLOUD comment='Cloud provider for outbound data transfer.',
		QUERY_HISTORY.OUTBOUND_DATA_TRANSFER_REGION as OUTBOUND_DATA_TRANSFER_REGION comment='Destination region for outbound data transfer.',
		QUERY_HISTORY.QUERY_HASH as QUERY_HASH comment='Hash value computed based on the canonicalized SQL text of the query.',
		QUERY_HISTORY.QUERY_ID as QUERY_ID comment='Unique identifier for the query.',
		QUERY_HISTORY.QUERY_PARAMETERIZED_HASH as QUERY_PARAMETERIZED_HASH comment='Hash value computed based on the parameterized query.',
		QUERY_HISTORY.QUERY_RETRY_CAUSE as QUERY_RETRY_CAUSE comment='Reason why the query was retried, if applicable.',
		QUERY_HISTORY.QUERY_TAG as QUERY_TAG comment='Query tag set for the query.',
		QUERY_HISTORY.QUERY_TEXT as QUERY_TEXT comment='SQL text of the query.',
		QUERY_HISTORY.QUERY_TYPE as QUERY_TYPE comment='SQL statement type (SELECT, INSERT, DELETE, etc.).',
		QUERY_HISTORY.RELEASE_VERSION as RELEASE_VERSION comment='Snowflake release version.',
		QUERY_HISTORY.ROLE_NAME as ROLE_NAME comment='Role that was active in the session at the time of the query.',
		QUERY_HISTORY.ROLE_TYPE as ROLE_TYPE comment='Type of role (e.g., SYSTEM_ROLE).',
		QUERY_HISTORY.SCHEMA_NAME as SCHEMA_NAME comment='Schema in which the SQL statement executed.',
		QUERY_HISTORY.SECONDARY_ROLE_STATS as SECONDARY_ROLE_STATS comment='JSON-formatted string containing information about secondary roles evaluated in the query.',
		QUERY_HISTORY.START_TIME as START_TIME comment='Time when the query started executing.',
		QUERY_HISTORY.USER_DATABASE_NAME as USER_DATABASE_NAME comment='Database name for Snowpark Container Services; otherwise NULL.',
		QUERY_HISTORY.USER_NAME as USER_NAME comment='User who issued the query.',
		QUERY_HISTORY.USER_SCHEMA_NAME as USER_SCHEMA_NAME comment='Schema name for Snowpark Container Services; otherwise NULL.',
		QUERY_HISTORY.USER_TYPE as USER_TYPE comment='Type of user executing the query (e.g., SNOWFLAKE_SERVICE for Container Services).',
		QUERY_HISTORY.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the warehouse used to run the query.',
		QUERY_HISTORY.WAREHOUSE_SIZE as WAREHOUSE_SIZE comment='Size of the warehouse used to run the query.',
		QUERY_HISTORY.WAREHOUSE_TYPE as WAREHOUSE_TYPE comment='Type of warehouse (STANDARD or SNOWPARK-OPTIMIZED).',
		QUERY_INSIGHTS.END_TIME as END_TIME comment='End time of the query that generated the insight.',
		QUERY_INSIGHTS.INSIGHT_INSTANCE_ID as INSIGHT_INSTANCE_ID comment='Unique identifier for the specific insight instance.',
		QUERY_INSIGHTS.INSIGHT_TOPIC as INSIGHT_TOPIC comment='General category or topic of the insight (e.g., performance, optimization).',
		QUERY_INSIGHTS.INSIGHT_TYPE_ID as INSIGHT_TYPE_ID comment='Specific type identifier for the insight within the topic.',
		QUERY_INSIGHTS.IS_OPPORTUNITY as IS_OPPORTUNITY comment='Boolean flag indicating whether the insight represents an optimization opportunity.',
		QUERY_INSIGHTS.MESSAGE as MESSAGE comment='Detailed message object containing the insight information and analysis.',
		QUERY_INSIGHTS.QUERY_HASH as QUERY_HASH comment='Hash value computed based on the canonicalized SQL text of the query.',
		QUERY_INSIGHTS.QUERY_ID as QUERY_ID comment='Unique identifier for the query that generated this insight.',
		QUERY_INSIGHTS.QUERY_PARAMETERIZED_HASH as QUERY_PARAMETERIZED_HASH comment='Hash value computed based on the parameterized query.',
		QUERY_INSIGHTS.START_TIME as START_TIME comment='Start time of the query that generated the insight.',
		QUERY_INSIGHTS.SUGGESTIONS as SUGGESTIONS comment='Array of actionable suggestions for query optimization.',
		QUERY_INSIGHTS.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the warehouse where the query was executed.',
		TABLE_PRUNING_HISTORY.DATABASE_NAME as DATABASE_NAME comment='Database that the table belongs to.',
		TABLE_PRUNING_HISTORY.END_TIME as END_TIME comment='End time for the time interval.',
		TABLE_PRUNING_HISTORY.SCHEMA_NAME as SCHEMA_NAME comment='Schema that the table belongs to.',
		TABLE_PRUNING_HISTORY.TABLE_NAME as TABLE_NAME comment='Name of the table.',
		TABLE_PRUNING_HISTORY.START_TIME as START_TIME comment='Start time for the time interval.',
		TABLE_QUERY_PRUNING_HISTORY.DATABASE_NAME as DATABASE_NAME comment='Database that the table belongs to.',
		TABLE_QUERY_PRUNING_HISTORY.INTERVAL_END_TIME as INTERVAL_END_TIME comment='End time for the time interval.',
		TABLE_QUERY_PRUNING_HISTORY.INTERVAL_START_TIME as INTERVAL_START_TIME comment='Start time for the time interval.',
		TABLE_QUERY_PRUNING_HISTORY.QUERY_HASH as QUERY_HASH comment='Hash value computed based on the canonicalized SQL text of the query.',
		TABLE_QUERY_PRUNING_HISTORY.QUERY_PARAMETERIZED_HASH as QUERY_PARAMETERIZED_HASH comment='Hash value computed based on the parameterized query.',
		TABLE_QUERY_PRUNING_HISTORY.SCHEMA_NAME as SCHEMA_NAME comment='Schema that the table belongs to.',
		TABLE_QUERY_PRUNING_HISTORY.TABLE_NAME as TABLE_NAME comment='Name of the table.',
		TABLE_QUERY_PRUNING_HISTORY.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the warehouse used to run the query.',
		WAREHOUSE_LOAD_HISTORY.END_TIME as END_TIME comment='End timestamp (UTC) of the time range during which the warehouse activity occurred.',
		WAREHOUSE_LOAD_HISTORY.START_TIME as START_TIME comment='Start timestamp (UTC) of the time range during which the warehouse activity occurred.',
		WAREHOUSE_LOAD_HISTORY.WAREHOUSE_NAME as WAREHOUSE_NAME comment='Name of the virtual warehouse.'
	)
	comment='This semantic view connects to key SNOWFLAKE.ACCOUNT_USAGE views that will allow us to analyze query history, query attribution history, query insights, and more. The goal is to serve insights about how we can optimize specific queries and workloads.'
	with extension (CA='{"tables":[{"name":"COLUMN_QUERY_PRUNING_HISTORY","dimensions":[{"name":"ACCESS_TYPE"},{"name":"COLUMN_NAME"},{"name":"DATABASE_NAME"},{"name":"QUERY_HASH"},{"name":"QUERY_PARAMETERIZED_HASH"},{"name":"SCHEMA_NAME"},{"name":"SEARCH_OPTIMIZATION_SUPPORTED_EXPRESSIONS"},{"name":"TABLE_NAME"},{"name":"VARIANT_PATH"},{"name":"WAREHOUSE_NAME"}],"facts":[{"name":"AGGREGATE_QUERY_COMPILATION_TIME"},{"name":"AGGREGATE_QUERY_ELAPSED_TIME"},{"name":"AGGREGATE_QUERY_EXECUTION_TIME"},{"name":"COLUMN_ID"},{"name":"DATABASE_ID"},{"name":"NUM_QUERIES"},{"name":"PARTITIONS_PRUNED"},{"name":"PARTITIONS_SCANNED"},{"name":"ROWS_MATCHED"},{"name":"ROWS_PRUNED"},{"name":"ROWS_SCANNED"},{"name":"SCHEMA_ID"},{"name":"TABLE_ID"},{"name":"WAREHOUSE_ID"}],"time_dimensions":[{"name":"INTERVAL_END_TIME"},{"name":"INTERVAL_START_TIME"}]},{"name":"QUERY_ACCELERATION_ELIGIBLE","dimensions":[{"name":"QUERY_HASH"},{"name":"QUERY_ID"},{"name":"QUERY_PARAMETERIZED_HASH"},{"name":"QUERY_TEXT"},{"name":"WAREHOUSE_NAME"},{"name":"WAREHOUSE_SIZE"}],"facts":[{"name":"ELIGIBLE_QUERY_ACCELERATION_TIME"},{"name":"QUERY_HASH_VERSION"},{"name":"QUERY_PARAMETERIZED_HASH_VERSION"},{"name":"UPPER_LIMIT_SCALE_FACTOR"}],"time_dimensions":[{"name":"END_TIME"},{"name":"START_TIME"}]},{"name":"QUERY_ATTRIBUTION_HISTORY","dimensions":[{"name":"PARENT_QUERY_ID"},{"name":"QUERY_HASH"},{"name":"QUERY_ID"},{"name":"QUERY_PARAMETERIZED_HASH"},{"name":"QUERY_TAG"},{"name":"ROOT_QUERY_ID"},{"name":"USER_NAME"},{"name":"WAREHOUSE_NAME"}],"facts":[{"name":"CREDITS_ATTRIBUTED_COMPUTE"},{"name":"CREDITS_USED_QUERY_ACCELERATION"},{"name":"WAREHOUSE_ID"}],"time_dimensions":[{"name":"END_TIME"},{"name":"START_TIME"}]},{"name":"QUERY_HISTORY","dimensions":[{"name":"DATABASE_NAME"},{"name":"ERROR_CODE"},{"name":"ERROR_MESSAGE"},{"name":"EXECUTION_STATUS"},{"name":"INBOUND_DATA_TRANSFER_CLOUD"},{"name":"INBOUND_DATA_TRANSFER_REGION"},{"name":"IS_CLIENT_GENERATED_STATEMENT"},{"name":"OUTBOUND_DATA_TRANSFER_CLOUD"},{"name":"OUTBOUND_DATA_TRANSFER_REGION"},{"name":"QUERY_HASH"},{"name":"QUERY_ID"},{"name":"QUERY_PARAMETERIZED_HASH"},{"name":"QUERY_RETRY_CAUSE"},{"name":"QUERY_TAG"},{"name":"QUERY_TEXT"},{"name":"QUERY_TYPE"},{"name":"RELEASE_VERSION"},{"name":"ROLE_NAME"},{"name":"ROLE_TYPE"},{"name":"SCHEMA_NAME"},{"name":"SECONDARY_ROLE_STATS"},{"name":"USER_DATABASE_NAME"},{"name":"USER_NAME"},{"name":"USER_SCHEMA_NAME"},{"name":"USER_TYPE"},{"name":"WAREHOUSE_NAME"},{"name":"WAREHOUSE_SIZE"},{"name":"WAREHOUSE_TYPE"}],"facts":[{"name":"BYTES_DELETED"},{"name":"BYTES_READ_FROM_RESULT"},{"name":"BYTES_SCANNED"},{"name":"BYTES_SENT_OVER_THE_NETWORK"},{"name":"BYTES_SPILLED_TO_LOCAL_STORAGE"},{"name":"BYTES_SPILLED_TO_REMOTE_STORAGE"},{"name":"BYTES_WRITTEN"},{"name":"BYTES_WRITTEN_TO_RESULT"},{"name":"CHILD_QUERIES_WAIT_TIME"},{"name":"CLUSTER_NUMBER"},{"name":"COMPILATION_TIME"},{"name":"CREDITS_USED_CLOUD_SERVICES"},{"name":"DATABASE_ID"},{"name":"EXECUTION_TIME"},{"name":"EXTERNAL_FUNCTION_TOTAL_INVOCATIONS"},{"name":"EXTERNAL_FUNCTION_TOTAL_RECEIVED_BYTES"},{"name":"EXTERNAL_FUNCTION_TOTAL_RECEIVED_ROWS"},{"name":"EXTERNAL_FUNCTION_TOTAL_SENT_BYTES"},{"name":"EXTERNAL_FUNCTION_TOTAL_SENT_ROWS"},{"name":"FAULT_HANDLING_TIME"},{"name":"INBOUND_DATA_TRANSFER_BYTES"},{"name":"LIST_EXTERNAL_FILES_TIME"},{"name":"OUTBOUND_DATA_TRANSFER_BYTES"},{"name":"PARTITIONS_SCANNED"},{"name":"PARTITIONS_TOTAL"},{"name":"PERCENTAGE_SCANNED_FROM_CACHE"},{"name":"QUERY_ACCELERATION_BYTES_SCANNED"},{"name":"QUERY_ACCELERATION_PARTITIONS_SCANNED"},{"name":"QUERY_ACCELERATION_UPPER_LIMIT_SCALE_FACTOR"},{"name":"QUERY_HASH_VERSION"},{"name":"QUERY_LOAD_PERCENT"},{"name":"QUERY_PARAMETERIZED_HASH_VERSION"},{"name":"QUERY_RETRY_TIME"},{"name":"QUEUED_OVERLOAD_TIME"},{"name":"QUEUED_PROVISIONING_TIME"},{"name":"QUEUED_REPAIR_TIME"},{"name":"ROWS_DELETED"},{"name":"ROWS_INSERTED"},{"name":"ROWS_PRODUCED"},{"name":"ROWS_UNLOADED"},{"name":"ROWS_UPDATED"},{"name":"ROWS_WRITTEN_TO_RESULT"},{"name":"SCHEMA_ID"},{"name":"SESSION_ID"},{"name":"TOTAL_ELAPSED_TIME"},{"name":"TRANSACTION_BLOCKED_TIME"},{"name":"TRANSACTION_ID"},{"name":"USER_DATABASE_ID"},{"name":"USER_SCHEMA_ID"},{"name":"WAREHOUSE_ID"}],"time_dimensions":[{"name":"END_TIME"},{"name":"START_TIME"}]},{"name":"QUERY_INSIGHTS","dimensions":[{"name":"INSIGHT_INSTANCE_ID"},{"name":"INSIGHT_TOPIC"},{"name":"INSIGHT_TYPE_ID"},{"name":"IS_OPPORTUNITY"},{"name":"MESSAGE"},{"name":"QUERY_HASH"},{"name":"QUERY_ID"},{"name":"QUERY_PARAMETERIZED_HASH"},{"name":"SUGGESTIONS"},{"name":"WAREHOUSE_NAME"}],"facts":[{"name":"TOTAL_ELAPSED_TIME"},{"name":"WAREHOUSE_ID"}],"time_dimensions":[{"name":"END_TIME"},{"name":"START_TIME"}]},{"name":"TABLE_PRUNING_HISTORY","dimensions":[{"name":"DATABASE_NAME"},{"name":"SCHEMA_NAME"},{"name":"TABLE_NAME"}],"facts":[{"name":"DATABASE_ID"},{"name":"NUM_SCANS"},{"name":"PARTITIONS_PRUNED"},{"name":"PARTITIONS_SCANNED"},{"name":"ROWS_PRUNED"},{"name":"ROWS_SCANNED"},{"name":"SCHEMA_ID"},{"name":"TABLE_ID"}],"time_dimensions":[{"name":"END_TIME"},{"name":"START_TIME"}]},{"name":"TABLE_QUERY_PRUNING_HISTORY","dimensions":[{"name":"DATABASE_NAME"},{"name":"QUERY_HASH"},{"name":"QUERY_PARAMETERIZED_HASH"},{"name":"SCHEMA_NAME"},{"name":"TABLE_NAME"},{"name":"WAREHOUSE_NAME"}],"facts":[{"name":"AGGREGATE_QUERY_COMPILATION_TIME"},{"name":"AGGREGATE_QUERY_ELAPSED_TIME"},{"name":"AGGREGATE_QUERY_EXECUTION_TIME"},{"name":"DATABASE_ID"},{"name":"NUM_QUERIES"},{"name":"PARTITIONS_PRUNED"},{"name":"PARTITIONS_SCANNED"},{"name":"ROWS_MATCHED"},{"name":"ROWS_PRUNED"},{"name":"ROWS_SCANNED"},{"name":"SCHEMA_ID"},{"name":"TABLE_ID"},{"name":"WAREHOUSE_ID"}],"time_dimensions":[{"name":"INTERVAL_END_TIME"},{"name":"INTERVAL_START_TIME"}]},{"name":"WAREHOUSE_LOAD_HISTORY","dimensions":[{"name":"WAREHOUSE_NAME"}],"facts":[{"name":"AVG_BLOCKED"},{"name":"AVG_QUEUED_LOAD"},{"name":"AVG_QUEUED_PROVISIONING"},{"name":"AVG_RUNNING"},{"name":"WAREHOUSE_ID"}],"time_dimensions":[{"name":"END_TIME"},{"name":"START_TIME"}]}],"relationships":[{"name":"QUERY_HISTORY_TO_QUERY_INSIGHTS"},{"name":"QUERY_HISTORY_TO_QUERY_ATTRIBUTION_HISTORY"}],"verified_queries":[{"name":"Show me the top 10 queries that would benefit from QAS","question":"Show me the top 10 queries that would benefit from QAS","sql":"SELECT\\n  query_id,\\n  query_hash,\\n  warehouse_name,\\n  warehouse_size,\\n  start_time,\\n  end_time,\\n  eligible_query_acceleration_time\\nFROM\\n  query_acceleration_eligible\\nORDER BY\\n  eligible_query_acceleration_time DESC NULLS LAST\\nLIMIT\\n  10","use_as_onboarding_question":false,"verified_by":"JCRITTENDEN","verified_at":1758141656}]}');

-- ============================================================================
-- [SECTION_6_BENCHMARK] BENCHMARK FUNCTIONS
-- ============================================================================

USE SCHEMA DIGITALSE.BENCHMARK;

-- Note: This function references a BENCHMARKS table that needs to be created separately
-- CREATE OR REPLACE FUNCTION fn_BENCHMARK_LOOKUP_BY_QUERY(QUERY_DESC_INPUT VARCHAR)
-- RETURNS NUMBER(38,9)
-- LANGUAGE SQL
-- AS
-- $$
-- (
--   SELECT AWS_DELTA_AWS_AVGS
--   FROM DIGITALSE.BENCHMARK.BENCHMARKS
--   WHERE AI_FILTER(
--           PROMPT('Return true if {0} contains similar SQL statements to {1}', QUERY_DESC_INPUT, QUERY_DESC)
--         ) = TRUE
--   ORDER BY AWS_DELTA_AWS_AVGS ASC
--   LIMIT 1
-- )
-- $$;

-- ============================================================================
-- [SECTION_7_HORIZON_CATALOG] AI CATALOG MANAGEMENT
-- ============================================================================

USE SCHEMA DIGITALSE.HORIZON_CATALOG;

-- Horizon Catalog Table for AI-Generated Descriptions
CREATE OR REPLACE TABLE DIGITALSE.HORIZON_CATALOG.horizon_catalog_descriptions (
  id INTEGER IDENTITY(1,1),
  domain VARCHAR(10) NOT NULL,
  name VARCHAR(256) NOT NULL,
  database_name VARCHAR(256) NOT NULL,
  schema_name VARCHAR(256) NOT NULL,
  table_name VARCHAR(256),
  description VARCHAR(16777216),
  description_version INTEGER DEFAULT 1,
  is_current BOOLEAN DEFAULT TRUE,
  is_applied_as_comment BOOLEAN DEFAULT FALSE,
  generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  applied_timestamp TIMESTAMP,
  source_metadata VARIANT,
  generation_source VARCHAR(50) DEFAULT 'AI_CORTEX',
  created_by STRING DEFAULT CURRENT_USER(),
  notes VARCHAR(1000)
) 
CLUSTER BY (database_name, schema_name, is_current);

-- View for current descriptions
CREATE OR REPLACE VIEW horizon_catalog_current_descriptions AS
SELECT 
  id,
  domain,
  name,
  database_name,
  schema_name,
  table_name,
  description,
  is_applied_as_comment,
  generation_timestamp,
  applied_timestamp,
  created_by
FROM DIGITALSE.HORIZON_CATALOG.horizon_catalog_descriptions
WHERE is_current = TRUE;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON DIGITALSE.HORIZON_CATALOG.horizon_catalog_descriptions TO ROLE PUBLIC;
GRANT SELECT ON DIGITALSE.HORIZON_CATALOG.horizon_catalog_current_descriptions TO ROLE PUBLIC;

/*
===============================================================================
HORIZON CATALOG DESCRIPTION GENERATION PROCEDURE
===============================================================================
*/

USE ROLE ACCOUNTADMIN;
USE SCHEMA DIGITALSE.HORIZON_CATALOG;

CREATE OR REPLACE PROCEDURE HORIZON_CATALOG_GENERATE_DESCRIPTIONS (
  database_name STRING, 
  schema_name STRING, 
  catalog_table STRING DEFAULT 'DIGITALSE.HORIZON_CATALOG.horizon_catalog_descriptions',
  force_regenerate BOOLEAN DEFAULT FALSE,
  include_sample_data BOOLEAN DEFAULT TRUE,
  max_parallel_jobs INTEGER DEFAULT NULL
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES=('snowflake-snowpark-python','joblib')
HANDLER = 'main'
AS
$$
import json
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime

def check_existing_descriptions(session, database_name, schema_name, table_name, catalog_table):
    """Check if descriptions already exist for a table and its columns"""
    try:
        # Check for existing table description
        table_query = f"""
        SELECT COUNT(*) as count 
        FROM {catalog_table} 
        WHERE database_name = '{database_name}' 
          AND schema_name = '{schema_name}' 
          AND name = '{table_name}' 
          AND domain = 'TABLE' 
          AND is_current = TRUE
        """
        table_result = session.sql(table_query).collect()
        has_table_desc = table_result[0]['COUNT'] > 0
        
        # Check for existing column descriptions
        column_query = f"""
        SELECT COUNT(*) as count 
        FROM {catalog_table} 
        WHERE database_name = '{database_name}' 
          AND schema_name = '{schema_name}' 
          AND table_name = '{table_name}' 
          AND domain = 'COLUMN' 
          AND is_current = TRUE
        """
        column_result = session.sql(column_query).collect()
        has_column_desc = column_result[0]['COUNT'] > 0
        
        return has_table_desc, has_column_desc
    except Exception as e:
        return False, False

def invalidate_old_descriptions(session, database_name, schema_name, table_name, catalog_table):
    """Mark existing descriptions as not current when regenerating"""
    try:
        # Invalidate old table description
        session.sql(f"""
        UPDATE {catalog_table} 
        SET is_current = FALSE 
        WHERE database_name = '{database_name}' 
          AND schema_name = '{schema_name}' 
          AND name = '{table_name}' 
          AND domain = 'TABLE' 
          AND is_current = TRUE
        """).collect()
        
        # Invalidate old column descriptions
        session.sql(f"""
        UPDATE {catalog_table} 
        SET is_current = FALSE 
        WHERE database_name = '{database_name}' 
          AND schema_name = '{schema_name}' 
          AND table_name = '{table_name}' 
          AND domain = 'COLUMN' 
          AND is_current = TRUE
        """).collect()
        
        return True
    except Exception as e:
        return False

def generate_descr(session, database_name, schema_name, table, catalog_table, force_regenerate, include_sample_data):
    """Generate descriptions for a single table and its columns"""
    table_name = table['TABLE_NAME']
    processing_result = {
        'table_name': table_name,
        'success': False,
        'message': '',
        'table_desc_generated': False,
        'column_desc_generated': False,
        'columns_processed': 0
    }
    
    try:
        # Check existing descriptions
        has_table_desc, has_column_desc = check_existing_descriptions(
            session, database_name, schema_name, table_name, catalog_table
        )
        
        # Skip if descriptions exist and not forcing regeneration
        if (has_table_desc and has_column_desc) and not force_regenerate:
            processing_result['message'] = 'Descriptions already exist, skipping'
            processing_result['success'] = True
            return processing_result
        
        # If regenerating, invalidate old descriptions
        if force_regenerate and (has_table_desc or has_column_desc):
            invalidate_old_descriptions(session, database_name, schema_name, table_name, catalog_table)
        
        # Generate new descriptions using AI
        ai_config = {
            'describe_columns': True, 
            'use_table_data': include_sample_data
        }
        
        async_job = session.sql(
            f"CALL AI_GENERATE_TABLE_DESC('{database_name}.{schema_name}.{table_name}', {ai_config})"
        ).collect_nowait()
        
        result = async_job.result()
        output = json.loads(result[0][0])
        columns_ret = output["COLUMNS"]
        table_ret = output["TABLE"][0]

        # Get next version number
        version_query = f"""
        SELECT COALESCE(MAX(description_version), 0) + 1 as next_version
        FROM {catalog_table} 
        WHERE database_name = '{database_name}' 
          AND schema_name = '{schema_name}' 
          AND name = '{table_name}' 
          AND domain = 'TABLE'
        """
        version_result = session.sql(version_query).collect()
        next_version = version_result[0]['NEXT_VERSION']

        # Process table description
        if not has_table_desc or force_regenerate:
            table_description = table_ret["description"].replace("'", "''")
            
            insert_table_sql = f"""
            INSERT INTO {catalog_table} (
                domain, name, database_name, schema_name, table_name, 
                description, description_version, is_current, 
                source_metadata, generation_timestamp
            ) VALUES (
                'TABLE', '{table_name}', '{database_name}', '{schema_name}', NULL,
                '{table_description}', {next_version}, TRUE,
                NULL, CURRENT_TIMESTAMP()
            )
            """
            session.sql(insert_table_sql).collect()
            processing_result['table_desc_generated'] = True

        # Process column descriptions
        columns_processed = 0
        if not has_column_desc or force_regenerate:
            for column in columns_ret:
                column_description = column["description"].replace("'", "''")
                column_name = column["name"]
                
                insert_column_sql = f"""
                INSERT INTO {catalog_table} (
                    domain, name, database_name, schema_name, table_name, 
                    description, description_version, is_current,
                    source_metadata, generation_timestamp
                ) VALUES (
                    'COLUMN', '{column_name}', '{database_name}', '{schema_name}', '{table_name}',
                    '{column_description}', {next_version}, TRUE,
                    NULL, CURRENT_TIMESTAMP()
                )
                """
                session.sql(insert_column_sql).collect()
                columns_processed += 1
            
            processing_result['column_desc_generated'] = True
            processing_result['columns_processed'] = columns_processed

        processing_result['success'] = True
        processing_result['message'] = f'Generated descriptions - Table: {processing_result["table_desc_generated"]}, Columns: {columns_processed}'
        
    except Exception as e:
        processing_result['message'] = f'Error: {str(e)}'
        
    return processing_result

def main(session, database_name, schema_name, catalog_table, force_regenerate, include_sample_data, max_parallel_jobs):
    """Main function to orchestrate description generation"""
    
    # Normalize inputs
    schema_name = schema_name.upper()
    database_name = database_name.upper()
    
    # Determine number of parallel jobs
    if max_parallel_jobs is None:
        max_parallel_jobs = multiprocessing.cpu_count()
    else:
        max_parallel_jobs = min(max_parallel_jobs, multiprocessing.cpu_count())
    
    # Get list of tables to process
    tables_query = f"""
    SELECT table_name
    FROM {database_name}.information_schema.tables
    WHERE table_schema = '{schema_name}'
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
    """
    
    try:
        tablenames = session.sql(tables_query).collect()
        
        if not tablenames:
            return f"No tables found in {database_name}.{schema_name}"
        
        # Process tables in parallel
        results = Parallel(n_jobs=max_parallel_jobs, backend="threading")(
            delayed(generate_descr)(
                session,
                database_name,
                schema_name,
                table,
                catalog_table,
                force_regenerate,
                include_sample_data
            ) for table in tablenames
        )
        
        # Compile summary
        total_tables = len(results)
        successful_tables = sum(1 for r in results if r['success'])
        total_table_descriptions = sum(1 for r in results if r['table_desc_generated'])
        total_column_descriptions = sum(r['columns_processed'] for r in results)
        
        # Log any failures
        failures = [r for r in results if not r['success']]
        
        summary = f"""Processing complete for {database_name}.{schema_name}:
- Tables processed: {total_tables}
- Successful: {successful_tables}
- Failed: {len(failures)}
- Table descriptions generated: {total_table_descriptions}
- Column descriptions generated: {total_column_descriptions}"""

        if failures:
            failure_details = "\\n".join([f"  - {f['table_name']}: {f['message']}" for f in failures])
            summary += f"\\n\\nFailures:\\n{failure_details}"
        
        return summary
        
    except Exception as e:
        return f"An error occurred: {str(e)}"
$$;

-- ============================================================================
-- INSTALLATION COMPLETE
-- ============================================================================

SELECT 
    '🎉 DIGITALSE COMPLETE INSTALLATION FINISHED' AS status,
    CURRENT_TIMESTAMP() AS completion_time,
    'Ready for use!' AS next_step;

-- Quick verification
SELECT 'Schemas created:' AS verification, COUNT(*) AS count 
FROM DIGITALSE.INFORMATION_SCHEMA.SCHEMATA;

SHOW PROCEDURES IN DATABASE DIGITALSE;

/*
===============================================================================
LOAD BENCHMARKS DATA - Gen1 vs Gen2 Warehouse Performance Comparison
===============================================================================

This script creates the BENCHMARKS table and loads Gen1 vs Gen2 warehouse
performance comparison data for the DigitalSE Workload Optimizer.

TABLE: DIGITALSE.BENCHMARK.BENCHMARKS

PURPOSE:
Store benchmark data comparing Gen1 and Gen2 warehouse performance across
various workload types (SCANS, JOINS, DML, SEMI-STRUCTURED, WINDOW functions).

Used by: FN_BENCHMARK_LOOKUP_BY_QUERY function to provide Gen2 performance
improvement estimates for query optimization recommendations.

EXECUTION TIME: < 1 minute

===============================================================================
*/

USE ROLE ACCOUNTADMIN;
USE DATABASE DIGITALSE;
USE SCHEMA BENCHMARK;

-- ============================================================================
-- CREATE BENCHMARKS TABLE
-- ============================================================================

CREATE OR REPLACE TABLE DIGITALSE.BENCHMARK.BENCHMARKS (
    -- Benchmark Metadata
    WAREHOUSE_NAME VARCHAR(100) COMMENT 'Name of the warehouse used for benchmarking',
    WORKLOAD VARCHAR(100) COMMENT 'Workload category (SCANS, JOINS, DML, SEMI-STRUCTURED, WINDOW)',
    QUERY_LABEL VARCHAR(100) COMMENT 'Specific query label within the workload',
    QUERIES NUMBER(10) COMMENT 'Number of queries in this benchmark',
    
    -- Gen1 Performance Metrics
    AVG_S NUMBER(18,9) COMMENT 'Gen1 average execution time in seconds',
    P50_S NUMBER(18,9) COMMENT 'Gen1 median (50th percentile) execution time in seconds',
    P90_S NUMBER(18,9) COMMENT 'Gen1 90th percentile execution time in seconds',
    P99_S NUMBER(18,9) COMMENT 'Gen1 99th percentile execution time in seconds',
    GB_SCANNED NUMBER(18,9) COMMENT 'Gen1 GB scanned',
    
    -- Gen2 Performance Metrics
    GEN2_AVG_S NUMBER(18,9) COMMENT 'Gen2 average execution time in seconds',
    GEN2_P50_S NUMBER(18,9) COMMENT 'Gen2 median (50th percentile) execution time in seconds',
    GEN2_P90_S NUMBER(18,9) COMMENT 'Gen2 90th percentile execution time in seconds',
    GEN2_P99_S NUMBER(18,9) COMMENT 'Gen2 99th percentile execution time in seconds',
    GEN2_GB_SCANNED NUMBER(18,9) COMMENT 'Gen2 GB scanned',
    
    -- Performance Improvement Deltas (negative = Gen2 faster)
    DELTA_AVG_S NUMBER(18,9) COMMENT 'Delta average time percentage (negative = Gen2 improvement)',
    AWS_DELTA_AWS_AVGS NUMBER(18,9) COMMENT 'AWS specific delta average percentage',
    GCP_DELTA_AVGS NUMBER(18,9) COMMENT 'GCP specific delta average percentage',
    AZURE_DELTA_AVGS NUMBER(18,9) COMMENT 'Azure specific delta average percentage',
    
    -- Query Text
    QUERY_DESC VARCHAR(16777216) COMMENT 'SQL query text for this benchmark',
    
    -- Metadata
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp',
    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record last update timestamp'
)
CLUSTER BY (WORKLOAD, WAREHOUSE_NAME)
COMMENT = 'Gen1 vs Gen2 warehouse performance benchmarks for query optimization recommendations';

-- ============================================================================
-- INSERT BENCHMARK DATA FROM CSV
-- ============================================================================

INSERT INTO DIGITALSE.BENCHMARK.BENCHMARKS (
    WAREHOUSE_NAME, WORKLOAD, QUERY_LABEL, QUERIES, 
    AVG_S, P50_S, P90_S, P99_S, GB_SCANNED,
    GEN2_AVG_S, GEN2_P50_S, GEN2_P90_S, GEN2_P99_S, GEN2_GB_SCANNED,
    DELTA_AVG_S, AWS_DELTA_AWS_AVGS, GCP_DELTA_AVGS, AZURE_DELTA_AVGS,
    QUERY_DESC
) VALUES

-- DML-Delete
('AICOLLEGE', 'DML-Delete', '3', 3, 2.476, 0.102, 5.82, 7.107, 0.016, 0.255, 0.057, 0.542, 0.651, 0.016, -89.70113086, -64.70113086, -64.70113086, -69.70113086,
'DELETE FROM fact_events WHERE event_type = ''support'' AND event_ts < DATEADD(''year'', -1, CURRENT_TIMESTAMP());'),

-- DML-Insert
('AICOLLEGE', 'DML-Insert', '1', 4, 0.704, 0.64, 1.388, 1.463, 0.109, 0.651, 0.55, 1.295, 1.406, 0.049, -7.528409091, 17.47159091, 17.47159091, 12.47159091,
'CREATE OR REPLACE TEMP TABLE tmp_sales_insert AS
SELECT * FROM fact_sales WHERE sale_date >= DATEADD(''day'', -30, CURRENT_DATE());

INSERT INTO fact_sales (sale_date, customer_id, product_id, quantity, unit_price, discount_pct, total_amount, channel, region, created_at, updated_at, attributes)
SELECT sale_date, customer_id, product_id, quantity, unit_price, discount_pct, total_amount, channel, region, DATEADD(''second'', 1, created_at), CURRENT_TIMESTAMP(), attributes
FROM tmp_sales_insert;'),

-- DML-Merge
('AICOLLEGE', 'DML-Merge', '4', 4, 0.68, 0.515, 1.452, 1.632, 0.003, 0.578, 0.575, 1.101, 1.106, 0.003, -15, 10, 10, 5,
'CREATE OR REPLACE TEMP TABLE tmp_dim_product_updates AS
SELECT product_id,
       category,
       subcategory,
       price * 1.01 AS price,
       active,
       OBJECT_INSERT(metadata, ''updated'', CURRENT_TIMESTAMP(), TRUE) AS metadata
FROM dim_product
QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM()) <= 2000;

MERGE INTO dim_product t
USING tmp_dim_product_updates s
ON t.product_id = s.product_id
WHEN MATCHED THEN UPDATE SET
  t.category = s.category,
  t.subcategory = s.subcategory,
  t.price = s.price,
  t.active = s.active,
  t.metadata = s.metadata
WHEN NOT MATCHED THEN INSERT (product_id, category, subcategory, price, active, metadata)
VALUES (s.product_id, s.category, s.subcategory, s.price, s.active, s.metadata);'),

-- DML-Transaction
('AICOLLEGE', 'DML-Transaction', '5', 4, 0.314, 0.313, 0.415, 0.451, 0.054, 0.305, 0.248, 0.507, 0.594, 0.054, -2.866242038, 22.13375796, 22.13375796, 17.13375796,
'BEGIN;
  UPDATE dim_customer SET status = ''SUSPENDED'' WHERE status = ''INACTIVE'' AND signup_date < DATEADD(''year'', -5, CURRENT_DATE());
  DELETE FROM fact_sales WHERE sale_date < DATEADD(''year'', -10, CURRENT_DATE());
COMMIT;'),

-- DML-Update
('AICOLLEGE', 'DML-Update', '2', 3, 0.336, 0.078, 0.717, 0.861, 0.036, 0.5, 0.089, 1.091, 1.317, 0.049, 48.80952381, 73.80952381, 73.80952381, 68.80952381,
'UPDATE fact_sales
SET discount_pct = LEAST(discount_pct + 1, 50),
    total_amount = ROUND(quantity * unit_price * (1 - LEAST(discount_pct + 1, 50)/100.0), 2),
    updated_at = CURRENT_TIMESTAMP()
WHERE sale_date >= DATEADD(''day'', -90, CURRENT_DATE())
  AND region IN (''NA'',''EMEA'');'),

-- JOINS-Exploded
('AICOLLEGE', 'JOINS-Exploded', '4', 1, 0.916, 0.916, 0.916, 0.916, 0.422, 0.597, 0.597, 0.597, 0.597, 0.422, -34.82532751, -9.825327511, -9.825327511, -14.82532751,
'WITH exploded AS (
  SELECT s.*, (UNIFORM(0, 9, RANDOM())) AS shard
  FROM fact_sales s
)
SELECT shard, COUNT(*) AS cnt, ROUND(SUM(total_amount),2) AS revenue
FROM exploded e
JOIN dim_customer c ON e.customer_id = c.customer_id
GROUP BY shard
ORDER BY shard;'),

-- JOINS-Rollup
('AICOLLEGE', 'JOINS-Rollup', '1', 3, 0.07, 0.07, 0.085, 0.089, 0, 0.083, 0.077, 0.1, 0.105, 0, 18.57142857, 43.57142857, 43.57142857, 38.57142857,
'SELECT d.year, d.month, p.category, p.subcategory, SUM(s.total_amount) AS revenue
FROM fact_sales s
JOIN dim_date d   ON s.sale_date = d.date_key
JOIN dim_product p ON s.product_id = p.product_id
GROUP BY ROLLUP(d.year, d.month, p.category, p.subcategory)
ORDER BY d.year, d.month, p.category, p.subcategory;'),

-- JOINS-Selective
('AICOLLEGE', 'JOINS-Selective', '2', 3, 0.108, 0.051, 0.193, 0.224, 0, 0.152, 0.08, 0.268, 0.31, 0, 40.74074074, 65.74074074, 65.74074074, 60.74074074,
'SELECT s.region, p.category, COUNT(*) AS orders, ROUND(SUM(s.total_amount),2) AS revenue
FROM fact_sales s
JOIN dim_product p ON s.product_id = p.product_id
WHERE p.active = TRUE AND s.sale_date >= DATEADD(''year'', -2, CURRENT_DATE())
GROUP BY s.region, p.category
ORDER BY revenue DESC;'),

-- JOINS-Semi-Structured
('AICOLLEGE', 'JOINS-Semi-Structured', '3', 3, 0.088, 0.066, 0.137, 0.153, 0, 0.068, 0.061, 0.08, 0.085, 0, -22.72727273, 2.272727273, 2.272727273, -2.727272727,
'SELECT p.category,
       COUNT(*) AS cnt,
       ROUND(AVG(TO_NUMBER(s.attributes:campaign_length::STRING, 10, 0)),2) AS avg_campaign_len
FROM (
  SELECT *, LENGTH(COALESCE(TO_VARCHAR(attributes:campaign), '''')) AS campaign_length
  FROM fact_sales
) s
JOIN dim_product p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY cnt DESC;'),

-- SCANS-Bloom
('AICOLLEGE', 'SCANS-Bloom', '5', 1, 0.541, 0.541, 0.541, 0.541, 0.119, 1.191, 1.191, 1.191, 1.191, 0.111, 120.1478743, 145.1478743, 145.1478743, 140.1478743,
'WITH hot_products AS (
  SELECT product_id FROM dim_product WHERE active = TRUE QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM()) <= 1000
)
SELECT SUM(total_amount)
FROM fact_sales s
JOIN hot_products hp USING(product_id)
WHERE s.sale_date >= DATEADD(''day'', -365, CURRENT_DATE());'),

-- SCANS-Clustered
('AICOLLEGE', 'SCANS-Clustered', '1', 3, 0.43, 0.183, 0.866, 1.02, 0.221, 0.063, 0.062, 0.066, 0.067, 0, -85.34883721, -60.34883721, -60.34883721, -65.34883721,
'SELECT region, COUNT(*) AS cnt
FROM fact_sales
GROUP BY region
ORDER BY cnt DESC;'),

-- SCANS-Date
('AICOLLEGE', 'SCANS-Date', '2', 3, 0.22, 0.082, 0.442, 0.523, 0.113, 0.067, 0.06, 0.078, 0.083, 0, -69.54545455, -44.54545455, -44.54545455, -49.54545455,
'SELECT sale_date, SUM(total_amount) AS revenue
FROM fact_sales
WHERE sale_date BETWEEN DATEADD(''day'', -90, CURRENT_DATE()) AND CURRENT_DATE()
GROUP BY sale_date
ORDER BY sale_date;'),

-- SCANS-Expressions
('AICOLLEGE', 'SCANS-Expressions', '3', 3, 0.236, 0.079, 0.469, 0.556, 0.221, 0.058, 0.059, 0.063, 0.064, 0, -75.42372881, -50.42372881, -50.42372881, -55.42372881,
'SELECT channel,
       IFF(discount_pct > 10, ''DISC10+'', ''DISC<10'') AS disc_band,
       COUNT(*) AS orders,
       ROUND(SUM(total_amount),2) AS revenue
FROM fact_sales
GROUP BY channel, disc_band
ORDER BY revenue DESC;'),

-- SCANS-Wide
('AICOLLEGE', 'SCANS-Wide', '4', 3, 0.518, 0.132, 1.106, 1.326, 0.693, 0.072, 0.07, 0.081, 0.084, 0, -86.1003861, -61.1003861, -61.1003861, -66.1003861,
'SELECT AVG(c001) AS avg1, AVG(c002) AS avg2, COUNT(*)
FROM wide_table
WHERE c051 > 100 AND c075 > 200;'),

-- SEMI-Flatten
('AICOLLEGE', 'SEMI-Flatten', '2', 3, 0.095, 0.072, 0.142, 0.158, 0, 0.115, 0.093, 0.16, 0.175, 0, 21.05263158, 46.05263158, 46.05263158, 41.05263158,
'WITH e AS (
  SELECT d.doc_id, f.value:type::STRING AS event_type
  FROM json_docs d,
       LATERAL FLATTEN(input => d.doc:events) f
)
SELECT event_type, COUNT(*) AS cnt
FROM e
GROUP BY event_type
ORDER BY cnt DESC;'),

-- SEMI-Join
('AICOLLEGE', 'SEMI-Join', '3', 1, 0.092, 0.092, 0.092, 0.092, 0, 0.095, 0.095, 0.095, 0.095, 0, 3.260869565, 28.26086957, 28.26086957, 23.26086957,
'WITH users AS (
  SELECT DISTINCT doc:user:id::NUMBER AS user_id
  FROM json_docs
)
SELECT COUNT(*) AS sales_to_json_users
FROM fact_sales s
JOIN users u ON s.customer_id = u.user_id;'),

-- SEMI-Select
('AICOLLEGE', 'SEMI-Select', '1', 3, 0.06, 0.06, 0.072, 0.075, 0, 0.056, 0.057, 0.063, 0.065, 0, -6.666666667, 18.33333333, 18.33333333, 13.33333333,
'SELECT doc:user:id::NUMBER AS user_id,
       doc:user:region::STRING AS region,
       ARRAY_SIZE(doc:tags) AS tag_count
FROM json_docs
WHERE doc:user:region::STRING IN (''NA'',''EMEA'');'),

-- WINDOW-Cohort-Rank
('AICOLLEGE', 'WINDOW-Cohort-Rank', '2', 3, 0.477, 0.123, 1.028, 1.231, 0.422, 0.103, 0.126, 0.131, 0.132, 0, -78.4067086, -53.4067086, -53.4067086, -58.4067086,
'WITH rev AS (
  SELECT c.signup_date, s.customer_id, SUM(s.total_amount) AS rev
  FROM fact_sales s
  JOIN dim_customer c ON s.customer_id = c.customer_id
  GROUP BY c.signup_date, s.customer_id
)
SELECT signup_date,
       customer_id,
       rev,
       RANK() OVER (PARTITION BY signup_date ORDER BY rev DESC) AS rnk,
       ROUND(PERCENT_RANK() OVER (PARTITION BY signup_date ORDER BY rev), 3) AS pct_rank
FROM rev
QUALIFY rnk <= 100
ORDER BY signup_date, rnk;'),

-- WINDOW-Daily
('AICOLLEGE', 'WINDOW-Daily', '1', 3, 0.153, 0.057, 0.298, 0.352, 0.221, 0.06, 0.064, 0.066, 0.066, 0, -60.78431373, -35.78431373, -35.78431373, -40.78431373,
'WITH daily AS (
  SELECT sale_date, SUM(total_amount) AS revenue
  FROM fact_sales
  GROUP BY sale_date
)
SELECT sale_date,
       revenue,
       ROUND(AVG(revenue) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS ma7
FROM daily
ORDER BY sale_date;'),

-- WINDOW-Sessionization
('AICOLLEGE', 'WINDOW-Sessionization', '3', 1, 1.234, 1.234, 1.234, 1.234, 0.292, 0.065, 0.065, 0.065, 0.065, 0, -94.73257699, -69.73257699, -69.73257699, -74.73257699,
'WITH ordered AS (
  SELECT customer_id,
         event_ts,
         LAG(event_ts) OVER (PARTITION BY customer_id ORDER BY event_ts) AS prev_ts
  FROM fact_events
)
SELECT customer_id,
       COUNT_IF(prev_ts IS NULL OR DATEDIFF(''minute'', prev_ts, event_ts) > 30) AS sessions_est
FROM ordered
GROUP BY customer_id
ORDER BY sessions_est DESC
LIMIT 1000;');

-- ============================================================================
-- VERIFICATION AND STATISTICS
-- ============================================================================

-- Verify record count
SELECT 'Benchmarks Loaded' AS status, COUNT(*) AS record_count
FROM DIGITALSE.BENCHMARK.BENCHMARKS;

-- Summary by workload category
SELECT 
    WORKLOAD,
    COUNT(*) AS query_count,
    ROUND(AVG(AWS_DELTA_AWS_AVGS), 2) AS avg_improvement_pct,
    ROUND(MIN(AWS_DELTA_AWS_AVGS), 2) AS best_improvement_pct,
    ROUND(MAX(AWS_DELTA_AWS_AVGS), 2) AS worst_improvement_pct
FROM DIGITALSE.BENCHMARK.BENCHMARKS
GROUP BY WORKLOAD
ORDER BY avg_improvement_pct;

-- Show queries with greatest Gen2 improvement (negative delta = improvement)
SELECT 
    WORKLOAD,
    QUERY_LABEL,
    ROUND(AVG_S, 3) AS gen1_avg_sec,
    ROUND(GEN2_AVG_S, 3) AS gen2_avg_sec,
    ROUND(AWS_DELTA_AWS_AVGS, 2) AS improvement_pct
FROM DIGITALSE.BENCHMARK.BENCHMARKS
ORDER BY AWS_DELTA_AWS_AVGS
LIMIT 10;

-- ============================================================================
-- UPDATE BENCHMARK FUNCTION
-- ============================================================================

-- Now create/update the benchmark lookup function
CREATE OR REPLACE FUNCTION DIGITALSE.BENCHMARK.FN_BENCHMARK_LOOKUP_BY_QUERY(QUERY_DESC_INPUT VARCHAR)
RETURNS NUMBER(38,9)
LANGUAGE SQL
COMMENT = 'Benchmark lookup function for Gen2 warehouse performance comparison. Returns expected performance improvement score based on similar query patterns.'
AS
$$
(
  SELECT AWS_DELTA_AWS_AVGS
  FROM DIGITALSE.BENCHMARK.BENCHMARKS
  WHERE AI_FILTER(
          PROMPT('Return true if {0} contains similar SQL statements to {1}', QUERY_DESC_INPUT, QUERY_DESC)
        ) = TRUE
  ORDER BY AWS_DELTA_AWS_AVGS ASC
  LIMIT 1
)
$$;

-- Test the benchmark function with a sample query
SELECT 
    'Benchmark Function Test' AS test_name,
    DIGITALSE.BENCHMARK.FN_BENCHMARK_LOOKUP_BY_QUERY(
        'SELECT region, COUNT(*) FROM sales GROUP BY region'
    ) AS expected_improvement_pct;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

SELECT 
    '✅ BENCHMARKS LOADED SUCCESSFULLY' AS status,
    (SELECT COUNT(*) FROM DIGITALSE.BENCHMARK.BENCHMARKS) AS total_records,
    'Function FN_BENCHMARK_LOOKUP_BY_QUERY ready for use' AS function_status,
    CURRENT_TIMESTAMP() AS completion_time;

/*
===============================================================================
USAGE EXAMPLES
===============================================================================

-- Find benchmarks for DML operations
SELECT * FROM DIGITALSE.BENCHMARK.BENCHMARKS 
WHERE WORKLOAD LIKE 'DML%'
ORDER BY AWS_DELTA_AWS_AVGS;

-- Find benchmarks with best Gen2 improvements
SELECT WORKLOAD, QUERY_LABEL, AWS_DELTA_AWS_AVGS
FROM DIGITALSE.BENCHMARK.BENCHMARKS
WHERE AWS_DELTA_AWS_AVGS < -50  -- More than 50% improvement
ORDER BY AWS_DELTA_AWS_AVGS;

-- Use the benchmark function
SELECT DIGITALSE.BENCHMARK.FN_BENCHMARK_LOOKUP_BY_QUERY(
    'SELECT * FROM fact_sales WHERE sale_date > CURRENT_DATE() - 90'
) AS expected_gen2_improvement;

===============================================================================
*/

