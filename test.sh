#!/bin/bash
mkdir -p test_data
cd test_data

cat > flow_correlation_events.csv << 'TESTEOF'
event_uuid,origin_addr,destination_addr,transport_proto,session_bytes,correlation_metadata,collection_source
FLOW_8847291,172.16.47.192:49152,crm-db01.internal.corp:3306,TCP_STREAM,2048576,database_query_session,netflow_collector_v9
FLOW_8847292,10.244.5.23:6443,172.31.0.55:3306,TCP_DATA,512000,k8s_api_to_mysql,envoy_telemetry
FLOW_8847293,192.168.100.45:443,203.0.113.50:80,HTTP_GET,1024000,web_content_request,panos_traffic_log
FLOW_8847294,172.16.0.0/12,backup-partner.external.net:22,SFTP_BULK,4194304,offsite_backup_transfer,cp_fw_logs
FLOW_8847295,mobile-pool.corp:993,exchange01.mail.corp:993,IMAPS_SYNC,65536,mobile_email_sync,zscaler_proxy
FLOW_8847296,jenkins.build.corp:8080,nexus.artifacts.corp:443,HTTPS_PUT,8388608,artifact_deployment,bigip_analytics
TESTEOF

cat > enterprise_inventory_data.csv << 'TESTEOF'
asset_reference,primary_fqdn,interface_config,platform_build,dept_assignment,facility_code,protection_agent
REF-001947,mailsvr-exch-01.corp.local,eth0:172.20.1.50;mgmt:192.168.1.50,win2019-std-build-v4.2,corp-it-messaging,facility-nyc-dc1,mde-sensor-v2.1
REF-002103,build-agent-07.dev.local,ens3:10.1.5.88;docker0:172.17.0.1,ubuntu-20.04-dev-template,software-eng-platform,cloud-aws-use1,qualys-agent-v4.8
REF-002456,analytics-node-01.data.corp,bond0:172.31.5.100;backup:192.168.100.100,rhel8-db-optimized-v2.1,data-intelligence-team,dc-chicago-tier3,tanium-client-v7.6
REF-002891,exec-laptop-pool-mgmt,wifi:192.168.50.0/24;vpn:10.200.0.0/24,macos-monterey-corp-image,executive-staff-support,distributed-mobile,s1-agent-v22.3
REF-003245,orchestrator-master-03.k8s,cni:10.244.0.0/16;flannel:10.100.0.0/16,container-os-optimized-v2023,platform-reliability-eng,gcp-central1-zone-c,falco-sensor-v0.35
REF-003672,perimeter-fw-01.ops.corp,mgmt:192.168.200.1;ext:203.0.113.1;int:10.0.0.1,cisco-asa-fw-9.18,network-sec-operations,colo-london-cage12,amp-connector-v1.8
TESTEOF

cat > endpoint_security_telemetry.csv << 'TESTEOF'
sensor_id,machine_identity,os_build,process_tree,file_hashes,registry_changes,agent_status
UUID-A7B9C3D1,CORP-LAPTOP-0447,Darwin-21.6.0-arm64,launchd|chrome.exe,sha256:4a5b6c7d8e9f1234,/Library/Preferences/com.apple,defender-online-v4.18
UUID-B8C2E4F5,WEB-FARM-02,Linux-5.4.0-ubuntu-x86_64,systemd|nginx,md5:1a2b3c4d5e6f7890,/etc/systemd/system/nginx,crowdstrike-falcon-6.45
UUID-C9D3F6A8,DB-PROD-01,Windows-NT-10.0.17763-x64,services.exe|sqlservr.exe,sha1:9f8e7d6c5b4a3210,HKLM\System\SQL\Security,tanium-client-7.4.6
UUID-D4E7G9B2,K8S-WORKER-05,Container-Linux-CoreOS-2023.4.0,containerd|kubelet,sha256:f1e2d3c4b5a67890,/var/lib/kubelet/config.yaml,falco-agent-0.35.1
UUID-E5F8H3C6,MOBILE-MGMT-99,iOS-16.7.2-arm64,kernel_task|SpringBoard,sha256:7g8h9i1j2k3l4567,/System/Library/Frameworks,jamf-protect-2.8.1
UUID-F6G9I4D7,FW-PERIMETER-01,Cisco-IOS-15.1-M12a,init|iosprocess,md5:m4n5o6p7q8r90123,running-config|startup-config,cisco-amp-1.2.3
TESTEOF

cd ..
echo "test" > test_mode.flag
