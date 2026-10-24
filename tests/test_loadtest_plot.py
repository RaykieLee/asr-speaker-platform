from __future__ import annotations

import json

from loadtest.plot import plot_report, render_html


def example_report():
    return {
        "config": {"mode": "mixed", "concurrency_levels": [1, 2]},
        "resource_samples": [
            {"sampled_at": "2026-10-23T01:00:00+00:00", "scope": "server_pid", "cpu_percent": 25, "rss_bytes": 104857600},
            {"sampled_at": "2026-10-23T01:00:02+00:00", "scope": "server_pid", "cpu_percent": 60, "rss_bytes": 157286400},
            {"sampled_at": "2026-10-23T01:00:03+00:00", "scope": "server_pid", "cpu_percent": None, "rss_bytes": None},
            {"sampled_at": "2026-10-23T01:00:04+00:00", "scope": "server_pid", "cpu_percent": 30, "rss_bytes": 188743680},
        ],
        "sessions": [
            {"run_id": "c1-a1-live-1", "mode": "realtime", "completed": True, "error": "",
             "first_partial_seconds": 0.5, "first_final_segment_seconds": 1.4,
             "completion_after_stop_seconds": 1.2,
             "final_text": "PRIVATE_TRANSCRIPT_NEVER_RENDER", "queue_seconds": None},
            {"run_id": "c2-a1-live-1", "mode": "realtime", "completed": False, "error": "TimeoutError: test",
             "first_partial_seconds": None, "completion_after_stop_seconds": None},
            {"run_id": "c2-a1-live-2", "mode": "realtime", "completed": True, "error": "",
             "first_partial_seconds": 0.9, "completion_after_stop_seconds": 1.7},
            {"run_id": "c2-a1-file-1", "mode": "file", "completed": True, "error": "",
             "queue_seconds": 0.4, "processing_seconds": 2.5, "rtf": 0.25},
        ],
    }


def test_render_has_offline_cpu_memory_and_concurrency_charts_without_transcripts():
    html = render_html(example_report())

    assert '<svg' in html
    assert 'CPU 使用率' in html
    assert 'RSS 常驻内存' in html
    assert '成功率（按并发档位和模式）' in html
    assert '关键指标 P95' in html
    assert '首个 final 片段到达（会话开始后）' in html
    assert '不表示音频结束后的识别延迟' in html
    assert '50.0%' in html
    assert html.count('<polyline') == 2  # both resource curves break at the missing sample
    assert html.count('<circle') == 6  # markers show measured points; no curve crosses the missing sample
    assert 'PRIVATE_TRANSCRIPT_NEVER_RENDER' not in html
    assert '<script' not in html


def test_missing_resource_and_latency_values_are_not_drawn_as_zero():
    report = {
        "config": {"concurrency_levels": [3], "mode": "file"},
        "resource_samples": [],
        "sessions": [{"run_id": "c3-one-file-1", "mode": "file", "completed": None,
                      "queue_seconds": None, "processing_seconds": None}],
    }
    html = render_html(report)

    assert html.count('不可评估') >= 8
    assert '不可评估：没有可用采样数据' in html
    assert '成功率（按并发档位和模式）' in html


def test_chart_reads_legacy_final_text_timing_under_explicit_label():
    report = {
        "config": {"concurrency_levels": [1], "mode": "realtime"},
        "resource_samples": [],
        "sessions": [{"run_id": "c1-legacy-live", "mode": "realtime", "completed": True,
                      "final_text_seconds": 12.5}],
    }
    html = render_html(report)
    assert '首个 final 片段到达（会话开始后）' in html
    assert '>12.500</text>' in html


def test_plot_report_writes_standalone_html(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(example_report()), encoding="utf-8")
    target = tmp_path / "charts" / "load.html"

    written = plot_report(source, target)

    assert written == target
    page = target.read_text(encoding="utf-8")
    assert 'PRIVATE_TRANSCRIPT_NEVER_RENDER' not in page
    assert 'resource_samples' not in page
    assert '<svg' in page
