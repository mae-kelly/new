-- =====================================================
-- LOG VISIBILITY MEASUREMENT SQL QUERIES
-- Based on AO1 Log Visibility Measurement Requirements
-- =====================================================

-- 1. GLOBAL VIEW - Overall Asset Coverage
-- =====================================================

-- 1.1 Total Asset Coverage Percentage
SELECT 
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS global_coverage_percentage,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging
FROM "data"."main"."all_sources";

-- 1.2 Asset Coverage by Log Source
SELECT 
    'Chronicle' AS log_source,
    COUNT(DISTINCT chronicle_device_hostname) AS assets_covered,
    COUNT(DISTINCT chronicle_device_hostname) * 100.0 / 
        (SELECT COUNT(DISTINCT hostname) FROM "data"."main"."all_sources") AS coverage_percentage
FROM "data"."main"."all_sources"
WHERE chronicle_device_hostname IS NOT NULL

UNION ALL

SELECT 
    'CrowdStrike' AS log_source,
    COUNT(DISTINCT crowdstrike_device_hostname) AS assets_covered,
    COUNT(DISTINCT crowdstrike_device_hostname) * 100.0 / 
        (SELECT COUNT(DISTINCT hostname) FROM "data"."main"."all_sources") AS coverage_percentage
FROM "data"."main"."all_sources"
WHERE crowdstrike_device_hostname IS NOT NULL

UNION ALL

SELECT 
    'Splunk' AS log_source,
    COUNT(DISTINCT splunk_host) AS assets_covered,
    COUNT(DISTINCT splunk_host) * 100.0 / 
        (SELECT COUNT(DISTINCT hostname) FROM "data"."main"."all_sources") AS coverage_percentage
FROM "data"."main"."all_sources"
WHERE splunk_host IS NOT NULL;

-- 2. INFRASTRUCTURE TYPE VIEW
-- =====================================================

-- 2.1 Coverage by Infrastructure Type (On-Prem, Cloud, SaaS, API)
SELECT 
    COALESCE(infra_type, 'Unknown') AS infrastructure_type,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS coverage_percentage
FROM "data"."main"."all_sources"
GROUP BY infra_type
ORDER BY coverage_percentage DESC;

-- 2.2 Detailed Coverage by Infrastructure and Log Type
SELECT 
    COALESCE(infra_type, 'Unknown') AS infrastructure_type,
    COALESCE(chronicle_log_type, splunk_sourcetype, 'Unknown') AS log_type,
    COUNT(DISTINCT hostname) AS asset_count,
    COUNT(DISTINCT CASE WHEN chronicle_log_type IS NOT NULL THEN hostname END) AS chronicle_coverage,
    COUNT(DISTINCT CASE WHEN splunk_sourcetype IS NOT NULL THEN hostname END) AS splunk_coverage,
    COUNT(DISTINCT CASE WHEN crowdstrike_device_hostname IS NOT NULL THEN hostname END) AS crowdstrike_coverage
FROM "data"."main"."all_sources"
GROUP BY infra_type, COALESCE(chronicle_log_type, splunk_sourcetype)
ORDER BY infrastructure_type, asset_count DESC;

-- 3. REGIONAL AND COUNTRY VIEW
-- =====================================================

-- 3.1 Coverage by Region
SELECT 
    COALESCE(region, 'Unknown') AS region,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS coverage_percentage
FROM "data"."main"."all_sources"
GROUP BY region
ORDER BY coverage_percentage DESC;

-- 3.2 Coverage by Country
SELECT 
    COALESCE(country, 'Unknown') AS country,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS coverage_percentage
FROM "data"."main"."all_sources"
GROUP BY country
ORDER BY coverage_percentage DESC;

-- 4. SYSTEM CLASSIFICATION VIEW
-- =====================================================

-- 4.1 Coverage by Operating System
SELECT 
    COALESCE(os, 'Unknown') AS operating_system,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS coverage_percentage
FROM "data"."main"."all_sources"
GROUP BY os
ORDER BY coverage_percentage DESC;

-- 4.2 Coverage by Platform
SELECT 
    COALESCE(platform, 'Unknown') AS platform,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS coverage_percentage
FROM "data"."main"."all_sources"
GROUP BY platform
ORDER BY coverage_percentage DESC;

-- 4.3 Coverage by Asset Class
SELECT 
    COALESCE(class, 'Unknown') AS asset_class,
    COUNT(DISTINCT hostname) AS total_assets,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) AS assets_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
    COUNT(DISTINCT hostname) AS coverage_percentage
FROM "data"."main"."all_sources"
GROUP BY class
ORDER BY coverage_percentage DESC;

-- 5. LOG TYPE AND ROLE MAPPING ANALYSIS
-- =====================================================

-- 5.1 Network Role Coverage (from requirements table)
SELECT 
    'Network' AS role,
    log_type,
    COUNT(DISTINCT hostname) AS asset_count,
    COUNT(DISTINCT ip_address) AS unique_ips
FROM (
    SELECT 
        CASE 
            WHEN chronicle_log_type IN ('Firewall Traffic', 'IDS/IPS', 'NDR', 'Proxy', 'DNS', 'WAF') 
                THEN chronicle_log_type
            WHEN splunk_sourcetype LIKE '%firewall%' OR splunk_sourcetype LIKE '%proxy%' OR splunk_sourcetype LIKE '%dns%'
                THEN splunk_sourcetype
            ELSE 'Other'
        END AS log_type,
        hostname,
        ip_address
    FROM "data"."main"."all_sources"
    WHERE chronicle_log_type IN ('Firewall Traffic', 'IDS/IPS', 'NDR', 'Proxy', 'DNS', 'WAF')
       OR splunk_sourcetype LIKE '%firewall%' 
       OR splunk_sourcetype LIKE '%proxy%' 
       OR splunk_sourcetype LIKE '%dns%'
) network_logs
GROUP BY log_type
ORDER BY asset_count DESC;

-- 5.2 Endpoint Role Coverage
SELECT 
    'Endpoint' AS role,
    log_type,
    COUNT(DISTINCT hostname) AS asset_count,
    COUNT(DISTINCT CASE WHEN crowdstrike_device_hostname IS NOT NULL THEN hostname END) AS crowdstrike_coverage
FROM (
    SELECT 
        CASE 
            WHEN chronicle_log_type IN ('OS logs (WinEVT, Linux syslog)', 'EDR', 'DLP', 'FIM')
                THEN chronicle_log_type
            WHEN splunk_sourcetype LIKE '%windows%' OR splunk_sourcetype LIKE '%linux%' OR splunk_sourcetype LIKE '%endpoint%'
                THEN splunk_sourcetype
            ELSE 'Other'
        END AS log_type,
        hostname,
        crowdstrike_device_hostname
    FROM "data"."main"."all_sources"
    WHERE chronicle_log_type IN ('OS logs (WinEVT, Linux syslog)', 'EDR', 'DLP', 'FIM')
       OR splunk_sourcetype LIKE '%windows%'
       OR splunk_sourcetype LIKE '%linux%'
       OR splunk_sourcetype LIKE '%endpoint%'
       OR crowdstrike_device_hostname IS NOT NULL
) endpoint_logs
GROUP BY log_type
ORDER BY asset_count DESC;

-- 5.3 Cloud Role Coverage
SELECT 
    'Cloud' AS role,
    log_type,
    COUNT(DISTINCT hostname) AS asset_count
FROM (
    SELECT 
        CASE 
            WHEN chronicle_log_type IN ('Cloud Event', 'Cloud Load Balancer', 'Cloud Config', 'Theom', 'Wiz', 'Cloud Security')
                THEN chronicle_log_type
            WHEN splunk_sourcetype LIKE '%cloud%' OR splunk_sourcetype LIKE '%aws%' OR splunk_sourcetype LIKE '%azure%'
                THEN splunk_sourcetype
            ELSE 'Other'
        END AS log_type,
        hostname
    FROM "data"."main"."all_sources"
    WHERE chronicle_log_type IN ('Cloud Event', 'Cloud Load Balancer', 'Cloud Config', 'Theom', 'Wiz', 'Cloud Security')
       OR splunk_sourcetype LIKE '%cloud%'
       OR splunk_sourcetype LIKE '%aws%'
       OR splunk_sourcetype LIKE '%azure%'
) cloud_logs
GROUP BY log_type
ORDER BY asset_count DESC;

-- 6. VISIBILITY FACTOR METRICS
-- =====================================================

-- 6.1 URL/FQDN Coverage Analysis
SELECT 
    'URL/FQDN Coverage' AS metric,
    COUNT(DISTINCT fqdn) AS total_fqdns,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN fqdn END) AS fqdns_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN fqdn END) * 100.0 / 
    COUNT(DISTINCT fqdn) AS coverage_percentage
FROM "data"."main"."all_sources"
WHERE fqdn IS NOT NULL;

-- 6.2 IP Address Coverage Analysis
SELECT 
    'IP Address Coverage' AS metric,
    COUNT(DISTINCT ip_address) AS total_ips,
    COUNT(DISTINCT CASE WHEN (chronicle_ip_address IS NOT NULL OR 
                             ip_address IS NOT NULL) THEN ip_address END) AS ips_with_logging,
    COUNT(DISTINCT CASE WHEN (chronicle_ip_address IS NOT NULL OR 
                             ip_address IS NOT NULL) THEN ip_address END) * 100.0 / 
    COUNT(DISTINCT ip_address) AS coverage_percentage
FROM "data"."main"."all_sources"
WHERE ip_address IS NOT NULL;

-- 7. LOG VOLUME AND INGEST ANALYSIS
-- =====================================================

-- 7.1 Log Ingest Volume by Source
SELECT 
    'Chronicle' AS source,
    SUM(COALESCE(chronicle_count, 0)) AS total_log_count,
    COUNT(DISTINCT chronicle_device_hostname) AS unique_hosts
FROM "data"."main"."all_sources"
WHERE chronicle_count IS NOT NULL

UNION ALL

SELECT 
    'Splunk' AS source,
    SUM(COALESCE(splunk_count, 0)) AS total_log_count,
    COUNT(DISTINCT splunk_host) AS unique_hosts
FROM "data"."main"."all_sources"
WHERE splunk_count IS NOT NULL

UNION ALL

SELECT 
    'Combined' AS source,
    SUM(COALESCE(log_count, 0)) AS total_log_count,
    COUNT(DISTINCT hostname) AS unique_hosts
FROM "data"."main"."all_sources"
WHERE log_count IS NOT NULL;

-- 8. CROWDSTRIKE AGENT HEALTH ANALYSIS
-- =====================================================

-- 8.1 CrowdStrike Agent Coverage and Health
SELECT 
    COALESCE(crowdstrike_agent_health, 'Unknown') AS agent_health,
    COALESCE(AgentStatus_Desc, 'Unknown') AS agent_status,
    COUNT(DISTINCT crowdstrike_device_hostname) AS agent_count,
    COUNT(DISTINCT crowdstrike_device_hostname) * 100.0 / 
        (SELECT COUNT(DISTINCT crowdstrike_device_hostname) 
         FROM "data"."main"."all_sources" 
         WHERE crowdstrike_device_hostname IS NOT NULL) AS percentage
FROM "data"."main"."all_sources"
WHERE crowdstrike_device_hostname IS NOT NULL
GROUP BY crowdstrike_agent_health, AgentStatus_Desc
ORDER BY agent_count DESC;

-- 9. GAPS AND ISSUES IDENTIFICATION
-- =====================================================

-- 9.1 Assets with No Logging Coverage
SELECT 
    hostname,
    fqdn,
    ip_address,
    COALESCE(region, 'Unknown') AS region,
    COALESCE(country, 'Unknown') AS country,
    COALESCE(infra_type, 'Unknown') AS infrastructure_type,
    COALESCE(os, 'Unknown') AS operating_system,
    COALESCE(class, 'Unknown') AS asset_class
FROM "data"."main"."all_sources"
WHERE chronicle_device_hostname IS NULL 
  AND crowdstrike_device_hostname IS NULL 
  AND splunk_host IS NULL
  AND hostname IS NOT NULL
ORDER BY region, country, infrastructure_type;

-- 9.2 Assets with Partial Coverage (missing from some sources)
SELECT 
    hostname,
    CASE WHEN chronicle_device_hostname IS NOT NULL THEN 'Yes' ELSE 'No' END AS chronicle_coverage,
    CASE WHEN crowdstrike_device_hostname IS NOT NULL THEN 'Yes' ELSE 'No' END AS crowdstrike_coverage,
    CASE WHEN splunk_host IS NOT NULL THEN 'Yes' ELSE 'No' END AS splunk_coverage,
    COALESCE(infra_type, 'Unknown') AS infrastructure_type,
    COALESCE(region, 'Unknown') AS region
FROM "data"."main"."all_sources"
WHERE hostname IS NOT NULL
  AND NOT (chronicle_device_hostname IS NOT NULL 
       AND crowdstrike_device_hostname IS NOT NULL 
       AND splunk_host IS NOT NULL)
  AND (chronicle_device_hostname IS NOT NULL 
    OR crowdstrike_device_hostname IS NOT NULL 
    OR splunk_host IS NOT NULL)
ORDER BY infrastructure_type, region;

-- 10. EXECUTIVE SUMMARY METRICS
-- =====================================================

-- 10.1 Key Performance Indicators
SELECT 
    'Total Assets' AS metric,
    COUNT(DISTINCT hostname) AS value,
    '%' AS unit
FROM "data"."main"."all_sources"

UNION ALL

SELECT 
    'Assets with Any Logging',
    COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                             crowdstrike_device_hostname IS NOT NULL OR 
                             splunk_host IS NOT NULL) THEN hostname END),
    'count'
FROM "data"."main"."all_sources"

UNION ALL

SELECT 
    'Overall Coverage Percentage',
    ROUND(COUNT(DISTINCT CASE WHEN (chronicle_device_hostname IS NOT NULL OR 
                                   crowdstrike_device_hostname IS NOT NULL OR 
                                   splunk_host IS NOT NULL) THEN hostname END) * 100.0 / 
          COUNT(DISTINCT hostname), 2),
    '%'
FROM "data"."main"."all_sources"

UNION ALL

SELECT 
    'Chronicle Coverage',
    ROUND(COUNT(DISTINCT chronicle_device_hostname) * 100.0 / 
          COUNT(DISTINCT hostname), 2),
    '%'
FROM "data"."main"."all_sources"

UNION ALL

SELECT 
    'CrowdStrike Coverage',
    ROUND(COUNT(DISTINCT crowdstrike_device_hostname) * 100.0 / 
          COUNT(DISTINCT hostname), 2),
    '%'
FROM "data"."main"."all_sources"

UNION ALL

SELECT 
    'Splunk Coverage',
    ROUND(COUNT(DISTINCT splunk_host) * 100.0 / 
          COUNT(DISTINCT hostname), 2),
    '%'
FROM "data"."main"."all_sources";

-- End of Queries