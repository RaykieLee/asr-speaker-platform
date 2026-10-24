"""Render a self-contained, offline HTML dashboard for a loadtest report."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_RUN_LEVEL = re.compile(r"^c(\d+)-")
_COLORS = ("#506bd8", "#27a37e", "#e59b3b", "#9a68c9", "#d05b64")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _polyline_chart(title: str, samples: list[dict[str, Any]], key: str,
                    unit: str, color: str, divisor: float = 1.0,
                    decimals: int = 1) -> str:
    width, height = 820, 250
    left, right, top, bottom = 58, 18, 34, 38
    plot_w, plot_h = width - left - right, height - top - bottom
    series = [(_timestamp(item.get("sampled_at")), _number(item.get(key))) for item in samples]
    timed = [(stamp, value / divisor) for stamp, value in series if stamp is not None and value is not None]
    heading = f'<text x="{left}" y="21" class="chart-title">{_esc(title)}</text>'
    if not timed:
        return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">'
                f'{heading}<rect class="empty-chart" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="8"/>'
                f'<text x="{left + plot_w/2:.1f}" y="{top + plot_h/2:.1f}" text-anchor="middle" class="empty-label">不可评估：没有可用采样数据</text></svg>')
    base_time = min(stamp for stamp, _ in timed)
    points = [(max(0.0, (stamp - base_time).total_seconds()), value) for stamp, value in timed]
    duration = max((x for x, _ in points), default=0.0)
    max_value = max((value for _, value in points), default=0.0)
    y_max = max_value * 1.12 if max_value > 0 else 1.0
    if key == "cpu_percent":
        y_max = max(100.0, y_max)
    x_ticks = 4
    y_ticks = 4
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">', heading]
    for idx in range(y_ticks + 1):
        fraction = idx / y_ticks
        y = top + plot_h * fraction
        value = y_max * (1 - fraction)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="axis-label">{value:.{decimals}f}</text>')
    for idx in range(x_ticks + 1):
        fraction = idx / x_ticks
        x = left + plot_w * fraction
        seconds = duration * fraction
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid vertical"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-12}" text-anchor="middle" class="axis-label">{seconds:.0f}s</text>')
    series_points: list[tuple[float, float] | None] = []
    for stamp, value in series:
        if stamp is None or value is None:
            series_points.append(None)
            continue
        seconds = max(0.0, (stamp - base_time).total_seconds())
        x = left + (seconds / duration * plot_w if duration > 0 else plot_w / 2)
        y = top + plot_h - ((value / divisor) / y_max * plot_h)
        series_points.append((x, y))
    # Missing samples split the curve instead of implying a measured trajectory.
    run: list[tuple[float, float]] = []

    def emit_run() -> None:
        if len(run) == 1:
            x, y = run[0]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        elif len(run) > 1:
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
            for x, y in run:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{color}"/>')

    for point in series_points:
        if point is None:
            emit_run()
            run = []
        else:
            run.append(point)
    emit_run()
    parts.append(f'<text x="{width-right}" y="{top+plot_h+24}" text-anchor="end" class="axis-label">{_esc(unit)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _session_groups(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, str, list[dict[str, Any]]]]]:
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    configured = config.get("concurrency_levels")
    levels: list[int] = [level for item in configured if isinstance(item, int) and not isinstance(item, bool)
                         for level in (item,)] if isinstance(configured, list) else []
    observed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for session in report.get("sessions", []) if isinstance(report.get("sessions"), list) else []:
        if not isinstance(session, dict):
            continue
        run_id = str(session.get("run_id", ""))
        match = _RUN_LEVEL.match(run_id)
        tier = match.group(1) if match else "未知并发"
        mode = str(session.get("mode") or "未知模式")
        observed.setdefault((tier, mode), []).append(session)
        if match and int(tier) not in levels:
            levels.append(int(tier))
    levels = sorted(set(levels))
    if not observed:
        mode = str(config.get("mode") or "未知模式")
        return levels, [(str(level), mode, []) for level in levels] or [("未知并发", mode, [])]
    keys = set(observed)
    for level in levels:
        if not any(tier == str(level) for tier, _ in keys):
            keys.add((str(level), str(config.get("mode") or "未知模式")))
    ordered = sorted(keys, key=lambda item: (0, int(item[0]), item[1]) if item[0].isdigit() else (1, item[0], item[1]))
    return levels, [(tier, mode, observed.get((tier, mode), [])) for tier, mode in ordered]


def _success_rate(sessions: list[dict[str, Any]]) -> float | None:
    if not sessions or any(not isinstance(session.get("completed"), bool) for session in sessions):
        return None
    successes = sum(bool(session.get("completed")) and not bool(session.get("error")) for session in sessions)
    return successes / len(sessions)


_LATENCIES = (
    ("首个 partial", "first_partial_seconds"),
    ("首个 final 片段到达（会话开始后）", "first_final_segment_seconds"),
    ("stop 后完成", "completion_after_stop_seconds"),
    ("文件排队", "queue_seconds"),
    ("文件处理", "processing_seconds"),
    ("RTF", "rtf"),
)
_P95_LATENCIES = tuple(item for item in _LATENCIES if item[1] != "rtf")


def _tier_chart(groups: list[tuple[str, str, list[dict[str, Any]]]]) -> str:
    width = 900
    row_h, top, left, right = 42, 45, 180, 95
    height = top + max(1, len(groups)) * row_h + 25
    plot_w = width - left - right
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="按并发档位统计成功率">',
             '<text x="12" y="23" class="chart-title">成功率（按并发档位和模式）</text>']
    for tick in range(0, 101, 25):
        x = left + plot_w * tick / 100
        parts.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{height-18}" class="grid vertical"/>')
        parts.append(f'<text x="{x:.1f}" y="{top-12}" text-anchor="middle" class="axis-label">{tick}%</text>')
    for index, (tier, mode, sessions) in enumerate(groups):
        y = top + index * row_h
        label = f"{tier} 路 · {mode}"
        parts.append(f'<text x="{left-12}" y="{y+17}" text-anchor="end" class="group-label">{_esc(label)}</text>')
        rate = _success_rate(sessions)
        if rate is None:
            parts.append(f'<text x="{left+4}" y="{y+17}" class="empty-label">不可评估（无成功状态数据）</text>')
        else:
            bar_w = plot_w * rate
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="23" rx="5" fill="#506bd8"/>')
            parts.append(f'<text x="{min(width-right-2, left+bar_w+8):.1f}" y="{y+16}" class="group-label">{rate*100:.1f}% · n={len(sessions)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _latency_chart(groups: list[tuple[str, str, list[dict[str, Any]]]]) -> str:
    def metric_value(session: dict[str, Any], key: str) -> float | None:
        value = session.get(key)
        if key == "first_final_segment_seconds" and value is None:
            # Read legacy reports, whose final_text_seconds has the same
            # session-start-to-first-final-event meaning.
            value = session.get("final_text_seconds")
        return _number(value)

    chart_groups = [(tier, mode, sessions, [(label, key, _percentile([value for session in sessions
                     if (value := metric_value(session, key)) is not None], 95))
                     for label, key in _P95_LATENCIES]) for tier, mode, sessions in groups]
    values = [value for _, _, _, metrics in chart_groups for _, _, value in metrics if value is not None]
    width, left, right = 1000, 230, 95
    row_h, top = 30, 44
    rows = sum(len(metrics) for _, _, _, metrics in chart_groups)
    height = max(180, top + rows * row_h + 24)
    plot_w = width - left - right
    x_max = max(values, default=0.0) * 1.1 or 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="按并发档位统计 P95 时延">',
             '<text x="12" y="23" class="chart-title">关键指标 P95（秒；RTF 为比值）</text>']
    for tick in range(5):
        frac = tick / 4
        x = left + plot_w * frac
        parts.append(f'<line x1="{x:.1f}" y1="{top-5}" x2="{x:.1f}" y2="{height-15}" class="grid vertical"/>')
        parts.append(f'<text x="{x:.1f}" y="{top-12}" text-anchor="middle" class="axis-label">{x_max*frac:.2f}</text>')
    y = top
    for tier, mode, _, metrics in chart_groups:
        for label, _, value in metrics:
            group_label = f"{tier}路 · {mode} · {label}"
            parts.append(f'<text x="{left-10}" y="{y+13}" text-anchor="end" class="group-label">{_esc(group_label)}</text>')
            if value is None:
                parts.append(f'<text x="{left+5}" y="{y+13}" class="empty-label">不可评估</text>')
            else:
                bar_w = max(1.5, plot_w * value / x_max)
                # Color by metric position for stable comparison across tiers.
                metric_index = next(i for i, item in enumerate(_P95_LATENCIES) if item[0] == label)
                color = _COLORS[metric_index % len(_COLORS)]
                parts.append(f'<rect x="{left}" y="{y-2}" width="{bar_w:.1f}" height="18" rx="4" fill="{color}"/>')
                parts.append(f'<text x="{min(width-right, left+bar_w+6):.1f}" y="{y+12}" class="group-label">{value:.3f}</text>')
            y += row_h
    parts.append("</svg>")
    return "".join(parts)


def _metrics_table(groups: list[tuple[str, str, list[dict[str, Any]]]]) -> str:
    def metric_value(session: dict[str, Any], key: str) -> float | None:
        value = session.get(key)
        if key == "first_final_segment_seconds" and value is None:
            value = session.get("final_text_seconds")
        return _number(value)

    rows = []
    for tier, mode, sessions in groups:
        success = _success_rate(sessions)
        count = len(sessions)
        cells = [f"{success*100:.1f}%" if success is not None else "不可评估"]
        for _, key in _LATENCIES:
            values = [value for session in sessions if (value := metric_value(session, key)) is not None]
            p50, p95 = _percentile(values, 50), _percentile(values, 95)
            cells.append(f"P50 {p50:.3f} / P95 {p95:.3f}" if p50 is not None and p95 is not None else "不可评估")
        rows.append(f'<tr><th>{_esc(tier)} 路 · {_esc(mode)}</th><td>{count if count else "不可评估"}</td>' +
                    "".join(f"<td>{_esc(cell)}</td>" for cell in cells) + "</tr>")
    headers = "".join(f"<th>{_esc(title)}</th>" for title, _ in _LATENCIES)
    return ("<div class=\"table-wrap\"><table><thead><tr><th>并发档位 / 模式</th><th>会话数</th><th>成功率</th>" +
            headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def render_html(report: dict[str, Any], source_name: str = "report.json") -> str:
    """Return offline HTML containing SVG charts and no session transcript text."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    samples = report.get("resource_samples") if isinstance(report.get("resource_samples"), list) else []
    levels, groups = _session_groups(report)
    scope_values = list(dict.fromkeys(str(item.get("scope")) for item in samples
                                     if isinstance(item, dict) and item.get("scope")))
    scope = ", ".join(scope_values) if scope_values else "采集范围未记录"
    config_levels = ", ".join(map(str, levels)) if levels else "未记录"
    resource_cpu = _polyline_chart("CPU 使用率", samples, "cpu_percent", "%", _COLORS[0])
    resource_rss = _polyline_chart("RSS 常驻内存", samples, "rss_bytes", "MiB", _COLORS[1], 1024 * 1024)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>负载测试报告图表</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6fa;color:#20283a;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:28px 22px 48px}}h1{{font-size:25px;margin:0 0 6px}}h2{{font-size:17px;margin:0 0 14px}}p,small{{color:#737e91}}.meta{{font-size:12px;margin:0 0 20px}}.card{{background:#fff;border:1px solid #e5e9f0;border-radius:13px;padding:17px;margin:14px 0;box-shadow:0 3px 12px #26365d08}}.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}svg{{display:block;width:100%;height:auto;overflow:visible}}.chart-title{{font-size:13px;font-weight:650;fill:#3b465b}}.axis-label,.empty-label{{font-size:10px;fill:#8993a4}}.group-label{{font-size:10px;fill:#526078}}.grid{{stroke:#e9edf3;stroke-width:1}}.grid.vertical{{stroke-dasharray:3 4}}.empty-chart{{fill:#fafbfe;stroke:#eef1f6}}table{{border-collapse:collapse;width:100%;font-size:11px;white-space:nowrap}}th,td{{padding:9px 10px;border-bottom:1px solid #edf0f4;text-align:left}}th{{color:#69758a;font-weight:600}}td{{color:#465269}}.table-wrap{{overflow:auto}}.note{{font-size:11px;line-height:1.7}}code{{background:#f1f3f7;border-radius:4px;padding:2px 5px}}@media(max-width:760px){{main{{padding:18px 12px}}.chart-grid{{grid-template-columns:1fr}}.card{{padding:13px}}}}
</style></head><body><main><h1>负载测试报告图表</h1><p class="meta">来源：{_esc(source_name)} · 模式：{_esc(config.get('mode') or '未记录')} · 并发档位：{_esc(config_levels)} · 资源范围：{_esc(scope)}</p>
<section class="card"><h2>服务资源随时间变化</h2><div class="chart-grid">{resource_cpu}{resource_rss}</div><p class="note">只绘制报告中有时间戳且有数值的采样点；缺失资源数据会显示为不可评估，不会补零。</p></section>
<section class="card"><h2>并发档位表现</h2>{_tier_chart(groups)}{_latency_chart(groups)}{_metrics_table(groups)}</section>
<section class="card note"><strong>指标口径</strong><p>成功率按各并发档位与负载模式内的会话数计算；成功表示 <code>completed=true</code> 且无错误。延迟显示可用样本的线性插值 P50/P95。首个 final 片段到达时间从会话开始计时，包含音频播放时长；它不表示音频结束后的识别延迟。报告未提供的指标会标为不可评估。图表不会展示会话的原始转写文本。</p></section></main></body></html>'''


def plot_report(input_path: Path, output_path: Path | None = None) -> Path:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report.json 顶层必须是 JSON object")
    destination = output_path or input_path.with_name("report.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(report, input_path.name), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Render loadtest report JSON as a self-contained offline HTML dashboard")
    parser.add_argument("report", type=Path, help="input report.json")
    parser.add_argument("-o", "--output", type=Path, help="output HTML path (default: report.html beside input)")
    args = parser.parse_args()
    try:
        output = plot_report(args.report, args.output)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(2, f"loadtest plot: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
