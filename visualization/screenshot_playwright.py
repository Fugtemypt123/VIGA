from playwright.sync_api import sync_playwright
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://0.0.0.0:8000/index.html", help="基础页面 URL，不带 dataset 参数")
parser.add_argument("--dataset-start", type=int, default=1, help="起始 dataset 序号（包含）")
parser.add_argument("--dataset-end", type=int, default=240, help="结束 dataset 序号（包含）")
parser.add_argument("--pdf-dir", default="visualization/pdf", help="PDF 输出目录")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fullpage", default=True)
parser.add_argument("--wait", default="networkidle", help="load/domcontentloaded/networkidle/timeout(ms)")
parser.add_argument("--delay", type=int, default=0, help="渲染后额外等待毫秒(动画/字体)")
parser.add_argument("--scale", type=float, default=2.0, help="deviceScaleFactor，1~3")
args = parser.parse_args()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        viewport={"width": args.width, "height": args.height},
        device_scale_factor=args.scale
    )
    page = ctx.new_page()

    base_url = args.url.split("?")[0]
    wait_until = args.wait if args.wait in ("load", "domcontentloaded", "networkidle") else "load"

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id in range(args.dataset_start, args.dataset_end + 1):
        target_url = f"{base_url}?dataset={dataset_id}"
        try:
            response = page.goto(target_url, wait_until=wait_until)
        except Exception as exc:
            print(f"[SKIP] dataset={dataset_id}: 页面访问失败 ({exc})")
            continue

        status = response.status if response else None
        if status and status >= 400:
            print(f"[SKIP] dataset={dataset_id}: HTTP {status}")
            continue

        if args.delay > 0:
            page.wait_for_timeout(args.delay)

        dataset_ready = page.evaluate("""
            () => {
                const fallback = document.querySelector('#generator_json');
                if (!fallback) return true;
                const style = window.getComputedStyle(fallback);
                if (style.display !== 'none') {
                    const text = (fallback.textContent || '').toLowerCase();
                    if (text.includes('not found') || text.includes('no datasets')) {
                        return false;
                    }
                }
                return true;
            }
        """)

        if not dataset_ready:
            print(f"[SKIP] dataset={dataset_id}: 页面提示数据不存在")
            continue

        pdf_path = pdf_dir / f"{dataset_id}.pdf"
        try:
            page.pdf(
                path=str(pdf_path),
                format="A4",
                margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"}
            )
            print(f"[OK] dataset={dataset_id}: {pdf_path}")
        except Exception as exc:
            print(f"[SKIP] dataset={dataset_id}: PDF 导出失败 ({exc})")

    browser.close()
