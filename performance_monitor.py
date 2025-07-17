import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

@dataclass
class OperationMetrics:
    operation_id: str
    operation_type: str
    start_time: float
    end_time: Optional[float] = None
    bytes_processed: int = 0
    rows_scanned: int = 0
    api_calls: int = 0
    error_count: int = 0
    success: bool = True
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def throughput_rows_per_second(self) -> float:
        duration = self.duration
        return self.rows_scanned / max(duration, 0.001)

    @property
    def throughput_bytes_per_second(self) -> float:
        duration = self.duration
        return self.bytes_processed / max(duration, 0.001)

class PerformanceMonitor:
    def __init__(self):
        self.active_operations: Dict[str, OperationMetrics] = {}
        self.completed_operations: List[OperationMetrics] = []
        self.operation_history: List[OperationMetrics] = []
        self.alerts: List[Dict] = []
        self.thresholds = {
            'slow_operation_seconds': 30,
            'low_throughput_rows_per_second': 100,
            'high_error_rate_percent': 10,
            'max_concurrent_operations': 50
        }

    def start_operation(self, operation_id: str, operation_type: str, resource_id: str = "") -> OperationMetrics:
        if len(self.active_operations) >= self.thresholds['max_concurrent_operations']:
            self._create_alert('high_concurrency', f"Too many concurrent operations: {len(self.active_operations)}")
        
        metrics = OperationMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=time.time(),
            resource_id=resource_id
        )
        
        self.active_operations[operation_id] = metrics
        return metrics

    def update_operation(self, operation_id: str, **kwargs):
        if operation_id in self.active_operations:
            metrics = self.active_operations[operation_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    if key in ['bytes_processed', 'rows_scanned', 'api_calls', 'error_count']:
                        setattr(metrics, key, getattr(metrics, key) + value)
                    else:
                        setattr(metrics, key, value)

    def end_operation(self, operation_id: str, success: bool = True, error_message: str = None):
        if operation_id in self.active_operations:
            metrics = self.active_operations[operation_id]
            metrics.end_time = time.time()
            metrics.success = success
            
            if error_message:
                metrics.details['error_message'] = error_message
                metrics.error_count += 1

            self.completed_operations.append(metrics)
            self.operation_history.append(metrics)
            del self.active_operations[operation_id]
            
            self._check_performance_thresholds(metrics)
            self._cleanup_old_history()

    def _check_performance_thresholds(self, metrics: OperationMetrics):
        if metrics.duration > self.thresholds['slow_operation_seconds']:
            self._create_alert('slow_operation', 
                f"Operation {metrics.operation_id} took {metrics.duration:.1f} seconds")

        if metrics.throughput_rows_per_second < self.thresholds['low_throughput_rows_per_second']:
            self._create_alert('low_throughput',
                f"Low throughput: {metrics.throughput_rows_per_second:.1f} rows/sec")

        recent_operations = self.operation_history[-20:]
        if len(recent_operations) >= 10:
            error_rate = sum(1 for op in recent_operations if not op.success) / len(recent_operations) * 100
            if error_rate > self.thresholds['high_error_rate_percent']:
                self._create_alert('high_error_rate', f"Error rate: {error_rate:.1f}%")

    def _create_alert(self, alert_type: str, message: str):
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if alert_type in ['high_error_rate', 'high_concurrency'] else 'medium'
        }
        self.alerts.append(alert)
        
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]

    def _cleanup_old_history(self):
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-500:]

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.completed_operations:
            return {'status': 'no_data', 'message': 'No completed operations to analyze'}

        recent_ops = self.completed_operations[-50:]
        
        durations = [op.duration for op in recent_ops]
        throughputs = [op.throughput_rows_per_second for op in recent_ops if op.rows_scanned > 0]
        success_rate = sum(1 for op in recent_ops if op.success) / len(recent_ops) * 100

        return {
            'summary_period': f"Last {len(recent_ops)} operations",
            'performance_metrics': {
                'average_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'p95_duration': self._percentile(durations, 95),
                'average_throughput_rows_per_sec': statistics.mean(throughputs) if throughputs else 0,
                'success_rate_percent': success_rate
            },
            'operation_breakdown': self._get_operation_breakdown(recent_ops),
            'resource_performance': self._get_resource_performance(recent_ops),
            'current_status': {
                'active_operations': len(self.active_operations),
                'recent_alerts': len([a for a in self.alerts if self._is_recent_alert(a)])
            }
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        return sorted_data[int(index)]

    def _get_operation_breakdown(self, operations: List[OperationMetrics]) -> Dict[str, Dict]:
        breakdown = {}
        
        for op in operations:
            op_type = op.operation_type
            if op_type not in breakdown:
                breakdown[op_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'success_count': 0,
                    'total_rows': 0
                }
            
            breakdown[op_type]['count'] += 1
            breakdown[op_type]['total_duration'] += op.duration
            breakdown[op_type]['total_rows'] += op.rows_scanned
            if op.success:
                breakdown[op_type]['success_count'] += 1

        for op_type, stats in breakdown.items():
            stats['average_duration'] = stats['total_duration'] / stats['count']
            stats['success_rate'] = (stats['success_count'] / stats['count']) * 100
            stats['average_rows_per_operation'] = stats['total_rows'] / stats['count']

        return breakdown

    def _get_resource_performance(self, operations: List[OperationMetrics]) -> Dict[str, Dict]:
        resource_stats = {}
        
        for op in operations:
            if not op.resource_id:
                continue
                
            if op.resource_id not in resource_stats:
                resource_stats[op.resource_id] = {
                    'operation_count': 0,
                    'total_duration': 0,
                    'error_count': 0,
                    'total_rows': 0
                }
            
            stats = resource_stats[op.resource_id]
            stats['operation_count'] += 1
            stats['total_duration'] += op.duration
            stats['total_rows'] += op.rows_scanned
            if not op.success:
                stats['error_count'] += 1

        for resource_id, stats in resource_stats.items():
            stats['average_duration'] = stats['total_duration'] / stats['operation_count']
            stats['error_rate'] = (stats['error_count'] / stats['operation_count']) * 100
            stats['throughput'] = stats['total_rows'] / stats['total_duration'] if stats['total_duration'] > 0 else 0

        return dict(sorted(resource_stats.items(), key=lambda x: x[1]['error_rate'], reverse=True)[:10])

    def _is_recent_alert(self, alert: Dict) -> bool:
        alert_time = datetime.fromisoformat(alert['timestamp'])
        return datetime.now() - alert_time < timedelta(hours=1)

    def get_performance_recommendations(self) -> List[str]:
        recommendations = []
        summary = self.get_performance_summary()
        
        if summary.get('status') == 'no_data':
            return ["Run some scan operations to generate performance recommendations"]

        metrics = summary['performance_metrics']
        
        if metrics['average_duration'] > 60:
            recommendations.append("Consider optimizing query patterns or increasing BigQuery slot allocation")

        if metrics['success_rate_percent'] < 90:
            recommendations.append("Investigate and resolve recurring operation failures")

        if metrics['average_throughput_rows_per_sec'] < 50:
            recommendations.append("Review sampling rates and query optimization opportunities")

        if len(self.active_operations) > 20:
            recommendations.append("Consider implementing operation queuing to manage concurrency")

        resource_perf = summary.get('resource_performance', {})
        high_error_resources = [rid for rid, stats in resource_perf.items() if stats['error_rate'] > 20]
        if high_error_resources:
            recommendations.append(f"Address connectivity issues with resources: {', '.join(high_error_resources[:3])}")

        recent_alerts = [a for a in self.alerts if self._is_recent_alert(a)]
        if len(recent_alerts) > 5:
            recommendations.append("Review recent performance alerts and implement corrective actions")

        return recommendations if recommendations else ["Performance looks good - no specific recommendations"]

    def export_performance_data(self) -> Dict[str, Any]:
        return {
            'export_timestamp': datetime.now().isoformat(),
            'active_operations': len(self.active_operations),
            'completed_operations_count': len(self.completed_operations),
            'performance_summary': self.get_performance_summary(),
            'recent_alerts': [a for a in self.alerts if self._is_recent_alert(a)],
            'recommendations': self.get_performance_recommendations(),
            'configuration': self.thresholds
        }

performance_monitor = PerformanceMonitor()
