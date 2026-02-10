"""
Prometheus metrics for POaaS.

Implements metrics endpoint at GET /metrics as specified in Appendix F.

Usage:
    from poaas.common.metrics import (
        request_latency, worker_calls, skip_rate,
        get_metrics_app
    )
    
    # Record metrics
    request_latency.labels(method="infer").observe(0.125)
    worker_calls.labels(worker="cleaner").inc()
    
    # Mount metrics endpoint
    app.mount("/metrics", get_metrics_app())
"""

import time
from functools import wraps
from typing import Callable

# Try to import prometheus_client, provide fallback if not available
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Route
    
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# Create custom registry for POaaS metrics
if HAS_PROMETHEUS:
    # Request latency histogram
    request_latency = Histogram(
        'poaas_request_latency_seconds',
        'Request latency in seconds',
        ['method', 'ablation'],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    
    # Worker call counter
    worker_calls = Counter(
        'poaas_worker_calls_total',
        'Total number of worker calls',
        ['worker', 'status']
    )
    
    # Skip rate gauge
    skip_counter = Counter(
        'poaas_skip_total',
        'Total number of skipped requests',
        ['reason']
    )
    
    processed_counter = Counter(
        'poaas_processed_total',
        'Total number of processed requests',
        ['ablation']
    )
    
    # Worker latency summary
    worker_latency = Summary(
        'poaas_worker_latency_seconds',
        'Worker latency in seconds',
        ['worker']
    )
    
    # Token counters
    tokens_in = Counter(
        'poaas_tokens_in_total',
        'Total input tokens processed',
        ['worker']
    )
    
    tokens_out = Counter(
        'poaas_tokens_out_total',
        'Total output tokens generated',
        ['worker']
    )
    
    # Drift gauge
    drift_histogram = Histogram(
        'poaas_drift_ratio',
        'Distribution of drift ratios',
        buckets=(0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50, 1.0)
    )
    
    # Length ratio histogram
    length_ratio_histogram = Histogram(
        'poaas_length_ratio',
        'Distribution of length expansion ratios',
        buckets=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.4, 3.0, 5.0)
    )
    
    # Quality score histogram
    quality_histogram = Histogram(
        'poaas_quality_score',
        'Distribution of input quality scores',
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
    )
    

def record_request(
    latency_seconds: float,
    method: str = "infer",
    ablation: str = "full",
    skipped: bool = False,
    skip_reason: str = ""
):
    """Record metrics for a request."""
    if not HAS_PROMETHEUS:
        return
    
    request_latency.labels(method=method, ablation=ablation).observe(latency_seconds)
    processed_counter.labels(ablation=ablation).inc()
    
    if skipped:
        skip_counter.labels(reason=skip_reason or "high_quality").inc()


def record_worker_call(
    worker: str,
    latency_seconds: float,
    success: bool = True,
    tokens_input: int = 0,
    tokens_output: int = 0
):
    """Record metrics for a worker call."""
    if not HAS_PROMETHEUS:
        return
    
    status = "success" if success else "error"
    worker_calls.labels(worker=worker, status=status).inc()
    worker_latency.labels(worker=worker).observe(latency_seconds)
    
    if tokens_input > 0:
        tokens_in.labels(worker=worker).inc(tokens_input)
    if tokens_output > 0:
        tokens_out.labels(worker=worker).inc(tokens_output)


def record_drift(drift_ratio: float, length_ratio: float):
    """Record drift and length ratio metrics."""
    if not HAS_PROMETHEUS:
        return
    
    drift_histogram.observe(drift_ratio)
    length_ratio_histogram.observe(length_ratio)


def record_quality(quality_score: float):
    """Record input quality score."""
    if not HAS_PROMETHEUS:
        return
    
    quality_histogram.observe(quality_score)


def get_metrics_response():
    """Generate Prometheus metrics response."""
    if not HAS_PROMETHEUS:
        return "# Prometheus client not installed\n", "text/plain"
    
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def get_metrics_app():
    """Get a Starlette app for the metrics endpoint."""
    if not HAS_PROMETHEUS:
        async def no_metrics(request):
            return Response(
                content="# Prometheus client not installed. Install with: pip install prometheus-client\n",
                media_type="text/plain"
            )
        return Starlette(routes=[Route("/", no_metrics)])
    
    async def metrics_endpoint(request):
        content, content_type = get_metrics_response()
        return Response(content=content, media_type=content_type)
    
    return Starlette(routes=[Route("/", metrics_endpoint)])


def timed_request(method: str = "infer"):
    """Decorator to time requests and record metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                latency = time.perf_counter() - start_time
                
                # Extract ablation from result if available
                ablation = "full"
                skipped = False
                if hasattr(result, 'ablation_mode'):
                    ablation = result.ablation_mode
                if hasattr(result, 'skipped'):
                    skipped = result.skipped
                
                record_request(latency, method, ablation, skipped)
                return result
            except Exception as e:
                latency = time.perf_counter() - start_time
                record_request(latency, method, "full", False)
                raise
        return wrapper
    return decorator


# Fallback implementations if prometheus_client not installed
if not HAS_PROMETHEUS:
    class NoOpMetric:
        """No-op metric for when prometheus_client is not installed."""
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass
        def inc(self, amount=1):
            pass
        def dec(self, amount=1):
            pass
        def set(self, value):
            pass
    
    request_latency = NoOpMetric()
    worker_calls = NoOpMetric()
    skip_counter = NoOpMetric()
    processed_counter = NoOpMetric()
    worker_latency = NoOpMetric()
    tokens_in = NoOpMetric()
    tokens_out = NoOpMetric()
    drift_histogram = NoOpMetric()
    length_ratio_histogram = NoOpMetric()
    quality_histogram = NoOpMetric()


if __name__ == "__main__":
    # Test metrics
    print(f"Prometheus available: {HAS_PROMETHEUS}")
    
    # Record some test metrics
    record_request(0.125, "infer", "full", False)
    record_worker_call("cleaner", 0.050, True, 50, 48)
    record_drift(0.12, 1.1)
    record_quality(0.85)
    
    # Generate metrics output
    content, content_type = get_metrics_response()
    print(f"\nMetrics ({content_type}):")
    if isinstance(content, bytes):
        print(content.decode('utf-8'))
    else:
        print(content)

