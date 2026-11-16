"""Smoke-test a running Signal Lab server through HTTP and an installed Chrome/Edge.

Usage: python scripts/smoke_web.py --url http://127.0.0.1:5000
No browser packages/downloads are required. The optional screenshot step uses the
websocket-client dependency supplied by Jupyter in requirements.txt. Browser data
lives in a new temporary profile, never in the user's personal browser profile.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def http_json(base_url: str, path: str, data: dict | None = None) -> tuple[int, dict]:
    body = None if data is None else urlencode(data).encode("utf-8")
    request = Request(base_url.rstrip("/") + path, data=body)
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def http_smoke(base_url: str) -> dict:
    status, health = http_json(base_url, "/api/health")
    assert status == 200 and health["status"] == "ok", health
    status, catalog = http_json(base_url, "/api/classes")
    assert status == 200 and len(catalog["classes"]) == 43, catalog
    status, models = http_json(base_url, "/api/models")
    assert status == 200, models
    status, samples = http_json(base_url, "/api/samples")
    assert status == 200, samples
    status, removed = http_json(base_url, "/api/benchmarks")
    assert status == 404 and "error" in removed, removed
    status, error = http_json(base_url, "/api/predict", {})
    assert status == 400 and isinstance(error.get("error"), str), error
    results = []
    ready = [model for model in models["models"] if model.get("ready")]
    if ready and samples["samples"]:
        sample = samples["samples"][0]
        for model in ready:
            status, result = http_json(base_url, "/api/predict", {
                "sample_id": sample["id"], "model_id": model["id"], "top_k": "5",
            })
            assert status == 200, result
            assert 0 <= result["prediction"]["class_id"] < 43, result
            assert result["model"]["id"] == model["id"], result
            assert 0 <= result["prediction"]["score"] <= 1, result
            assert result["processed_image"].startswith("data:image/png;base64,"), result
            assert len(result["top_k"]) == min(5, model.get("n_classes") or 43), result
            results.append({"model_id": model["id"], "class_id": result["prediction"]["class_id"],
                            "label": result["prediction"]["label"], "elapsed_ms": result["elapsed_ms"]})
        status, error = http_json(base_url, "/api/predict", {
            "sample_id": sample["id"], "roi": "[0,0,999999,999999]",
        })
        assert status == 400 and "error" in error, error
    return {"health": health, "classes": len(catalog["classes"]),
            "models": len(models["models"]), "samples": len(samples["samples"]),
            "removed_benchmarks": "404", "predictions": results}


class Browser:
    """Minimal Chrome DevTools Protocol client for this smoke test."""

    def __init__(self, executable: Path):
        import websocket

        self.profile = Path(tempfile.mkdtemp(prefix="signal-lab-browser-"))
        self.process = subprocess.Popen([
            str(executable), "--headless=new", "--remote-debugging-port=0",
            f"--user-data-dir={self.profile}", "--no-first-run", "--no-default-browser-check",
            "--disable-background-networking", "--disable-component-update", "--disable-gpu",
            "--disable-default-apps", "--disable-sync", "--disable-extensions",
            "--disable-features=Translate,MediaRouter,OptimizationHints,OptimizationGuideModelDownloading,OptimizationTargetPrediction",
            "--metrics-recording-only", "--no-pings", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws = None
        self.sequence = 0
        self.console_errors: list[str] = []
        try:
            deadline = time.monotonic() + 30
            active = self.profile / "DevToolsActivePort"
            while not active.is_file() and time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Browser exited before DevTools became available")
                time.sleep(0.1)
            port = int(active.read_text(encoding="utf-8").splitlines()[0])
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
                targets = json.load(response)
            target = next(item for item in targets if item["type"] == "page")
            self.ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=90,
                                                   suppress_origin=True)
            self.command("Page.enable")
            self.command("Runtime.enable")
            self.command("Log.enable")
        except Exception:
            self.close()
            raise

    def command(self, method: str, params: dict | None = None) -> dict:
        self.sequence += 1
        sequence = self.sequence
        self.ws.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                self.console_errors.append(str(message.get("params", {}).get("exceptionDetails", {})))
            if message.get("method") == "Log.entryAdded":
                entry = message.get("params", {}).get("entry", {})
                if entry.get("level") == "error":
                    self.console_errors.append(entry.get("text", "Browser log error"))
            if message.get("method") == "Runtime.consoleAPICalled":
                if message.get("params", {}).get("type") == "error":
                    self.console_errors.append(str(message["params"].get("args", [])))
            if message.get("id") == sequence:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression: str, *, user_gesture: bool = False):
        result = self.command("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True,
            "userGesture": user_gesture,
        })
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def wait(self, expression: str, seconds: int = 30):
        return self.evaluate("""new Promise((resolve,reject)=>{
            const end=Date.now()+%d;
            function check(){try{const value=(%s);if(value){resolve(value);return;}}
            catch(error){reject(String(error));return;}
            if(Date.now()>end){reject('Timed out waiting for browser state');return;}
            setTimeout(check,100);} check();})""" % (seconds * 1000, expression))

    def shortcut(self, key: str):
        """Send a real Ctrl shortcut through the browser input pipeline."""
        code = key.upper()
        self.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Control", "code": "ControlLeft", "windowsVirtualKeyCode": 17, "modifiers": 2})
        self.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": key.lower(), "code": "Key" + code, "windowsVirtualKeyCode": ord(code), "modifiers": 2})
        self.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": key.lower(), "code": "Key" + code, "windowsVirtualKeyCode": ord(code), "modifiers": 2})
        self.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Control", "code": "ControlLeft", "windowsVirtualKeyCode": 17})

    def screenshot(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        result = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(result["data"]))

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        shutil.rmtree(self.profile, ignore_errors=True)


def clipboard_smoke(browser: Browser) -> dict:
    """Copy the focused full input image and paste it through trusted shortcuts."""
    browser.command("Page.bringToFront")
    browser.command("Emulation.setFocusEmulationEnabled", {"enabled": True})
    browser.wait("document.hasFocus()")
    dimensions = browser.evaluate("({width:document.querySelector('#upload-preview').naturalWidth,height:document.querySelector('#upload-preview').naturalHeight})")
    browser.evaluate("navigator.clipboard.writeText('signal-lab-clipboard-smoke')", user_gesture=True)
    browser.evaluate("document.querySelector('#input-image-area').focus();window.getSelection().removeAllRanges()")
    browser.shortcut("c")
    browser.wait("document.querySelector('#clipboard-status').textContent.includes('Đã sao chép')")
    copied = browser.evaluate("""(async()=>{
        const items=await navigator.clipboard.read();
        const item=items.find(value=>value.types.includes('image/png'));
        if(!item)throw new Error('Clipboard has no PNG image');
        const bitmap=await createImageBitmap(await item.getType('image/png'));
        const size={width:bitmap.width,height:bitmap.height};bitmap.close();return size;
    })()""", user_gesture=True)
    assert copied == dimensions, (copied, dimensions)
    browser.evaluate("document.querySelector('#remove-image').click();document.querySelector('#image-file').focus()")
    browser.shortcut("v")
    browser.wait("!document.querySelector('#image-preview-panel').hidden && document.querySelector('#upload-preview').naturalWidth > 0 && !document.querySelector('#copy-image-button').disabled")
    pasted = browser.evaluate("({width:document.querySelector('#upload-preview').naturalWidth,height:document.querySelector('#upload-preview').naturalHeight})")
    assert pasted == dimensions, (pasted, dimensions)
    assert not browser.evaluate("document.querySelector('#roi-enabled').disabled"), "Pasted image should behave as a file upload"
    # Editing and selecting text must not overwrite the image clipboard.
    browser.evaluate("document.querySelector('#roi-details').open=true;document.querySelector('#roi-enabled').click();const field=document.querySelector('#roi-x1');field.focus();field.value='7'")
    browser.shortcut("a")
    browser.shortcut("c")
    assert browser.evaluate("navigator.clipboard.readText()", user_gesture=True) == "7", "Ctrl+C in ROI must copy text, not image"
    browser.evaluate("navigator.clipboard.writeText('2')", user_gesture=True)
    browser.shortcut("v")
    assert browser.evaluate("document.querySelector('#roi-x1').value") == "2", "Ctrl+V in ROI must paste text normally"
    browser.evaluate("document.querySelector('#roi-enabled').click()")
    browser.evaluate("navigator.clipboard.writeText('not-an-image')", user_gesture=True)
    browser.evaluate("document.querySelector('#main-content').focus()")
    browser.shortcut("v")
    assert browser.evaluate("!document.querySelector('#image-preview-panel').hidden"), "Non-image paste must preserve selected image"
    # A dedicated copy button uses the same real system clipboard path.
    browser.evaluate("document.querySelector('#copy-image-button').click()", user_gesture=True)
    browser.wait("document.querySelector('#clipboard-status').textContent.includes('Đã sao chép') && !document.querySelector('#copy-image-button').disabled")
    assert browser.evaluate("(async()=>{const rows=await navigator.clipboard.read();return rows.some(row=>row.types.includes('image/png'));})()", user_gesture=True)
    before = browser.evaluate("document.querySelector('#file-name').textContent")
    browser.evaluate("document.querySelector('[data-view=danh-muc]').click();document.querySelector('#main-content').focus()")
    browser.wait("!document.querySelector('#view-danh-muc').hidden")
    browser.shortcut("v")
    assert browser.evaluate("document.querySelector('#file-name').textContent") == before, "Catalog paste must not replace input"
    browser.evaluate("document.querySelector('[data-view=nhan-dien]').click()")
    browser.wait("!document.querySelector('#view-nhan-dien').hidden")
    return {"keyboard_copy": "passed", "keyboard_paste": "passed", "copy_button": "passed",
            "editable_text_shortcuts": "passed", "catalog_paste_ignored": "passed",
            "full_image_dimensions": dimensions}


def browser_smoke(base_url: str, output_dir: Path, browser_path: str | None,
                  upload_image: Path | None = None, upload_roi: list[int] | None = None) -> dict:
    choices = ([Path(browser_path)] if browser_path else [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
    ])
    executable = next((path for path in choices if path.is_file()), None)
    if executable is None:
        raise RuntimeError("No Chrome/Edge found; supply --browser or use --http-only")
    browser = Browser(executable)
    try:
        browser.command("Emulation.setDeviceMetricsOverride", {
            "width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False,
        })
        browser.command("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]})
        origin = "http://" + urlparse(base_url).netloc
        # Permission is limited to this disposable profile and local test origin.
        browser.command("Browser.grantPermissions", {"origin": origin, "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"]})
        browser.command("Page.navigate", {"url": base_url})
        browser.wait("document.readyState === 'complete' && document.querySelector('#model-select')")
        browser.wait("document.querySelector('#sample-list[aria-busy=false]') && !document.querySelector('#model-description').textContent.includes('Đang')")
        browser.wait("Array.from(document.querySelectorAll('#sample-list img')).every(img=>img.complete)")
        browser.screenshot(output_dir / "desktop.png")
        did_predict = False
        model_id = browser.evaluate("document.querySelector('#model-select').value")
        has_sample = browser.evaluate("Boolean(document.querySelector('[data-sample-id]'))")
        if has_sample:
            browser.evaluate("document.querySelector('[data-sample-id]').click()")
            browser.wait("document.querySelector('#upload-preview').complete && document.querySelector('#upload-preview').naturalWidth > 0")
            enabled = browser.evaluate("Boolean(document.querySelector('#recognize-button:not(:disabled)'))")
            if enabled:
                browser.evaluate("document.querySelector('#recognize-button').click()")
                browser.wait("document.querySelector('#result-panel[data-state=success]')", 60)
                browser.wait("document.querySelector('#predicted-icon').complete")
                bars = browser.evaluate("Array.from(document.querySelectorAll('.score-progress')).map(el=>({width:el.getBoundingClientRect().width,value:el.value,max:el.max}))")
                assert bars and any(bar["width"] > 0 and bar["value"] > 0 for bar in bars), bars
                browser.screenshot(output_dir / "prediction.png")
                browser.evaluate("document.querySelector('#score-table-toggle').click()")
                assert browser.evaluate("!document.querySelector('#top-table-wrapper').hidden")
                browser.evaluate("document.querySelector('#score-table-toggle').click()")
                did_predict = True
                assert browser.evaluate("document.querySelector('#roi-enabled').disabled"), "Samples must use CSV ROI"
        uploaded = False
        if upload_image is not None and model_id:
            document = browser.command("DOM.getDocument")
            field = browser.command("DOM.querySelector", {"nodeId": document["root"]["nodeId"], "selector": "#image-file"})
            browser.command("DOM.setFileInputFiles", {"nodeId": field["nodeId"], "files": [str(upload_image.resolve())]})
            browser.wait("!document.querySelector('#image-preview-panel').hidden && !document.querySelector('#recognize-button').disabled")
            assert not browser.evaluate("document.querySelector('#roi-enabled').disabled"), "Uploads allow ROI"
            if upload_roi is not None:
                browser.evaluate("document.querySelector('#roi-enabled').click()")
                browser.evaluate("const coords=" + json.dumps(upload_roi) + ";['roi-x1','roi-y1','roi-x2','roi-y2'].forEach((id,i)=>{const field=document.getElementById(id);field.value=String(coords[i]);field.dispatchEvent(new Event('input',{bubbles:true}));})")
                browser.wait("!document.querySelector('#recognize-button').disabled")
            browser.evaluate("document.querySelector('#recognize-button').click()")
            browser.wait("document.querySelector('#result-panel[data-state=success]')", 60)
            browser.screenshot(output_dir / "upload-prediction.png")
            uploaded = True
        assert browser.evaluate("!document.querySelector('[data-view=thuc-nghiem]') && !document.querySelector('#view-thuc-nghiem') && !document.querySelector('#download-result')"), "Removed features must not remain in UI"
        browser.evaluate("document.querySelector('[data-view=danh-muc]').click()")
        browser.wait("!document.querySelector('#view-danh-muc').hidden && document.querySelectorAll('.class-card').length === 43")
        browser.evaluate("document.querySelector('#class-search').value='Dừng lại';document.querySelector('#class-search').dispatchEvent(new Event('input',{bubbles:true}))")
        browser.wait("document.querySelectorAll('.class-card').length === 1")
        browser.wait("Array.from(document.querySelectorAll('.class-card img')).every(img=>img.complete)")
        browser.screenshot(output_dir / "catalog-search.png")
        browser.evaluate("document.querySelector('.skip-link').click()")
        assert browser.evaluate("!document.querySelector('#view-danh-muc').hidden"), "Skip link must preserve selected view"
        browser.evaluate("document.querySelector('[data-view=nhan-dien]').click()")
        browser.wait("!document.querySelector('#view-nhan-dien').hidden")
        browser.evaluate("document.querySelector('#theme-select').value='dark';document.querySelector('#theme-select').dispatchEvent(new Event('change',{bubbles:true}))")
        browser.wait("document.documentElement.dataset.theme === 'dark'")
        browser.screenshot(output_dir / "dark.png")
        browser.evaluate("document.querySelector('#theme-select').value='light';document.querySelector('#theme-select').dispatchEvent(new Event('change',{bubbles:true}))")
        browser.command("Emulation.setDeviceMetricsOverride", {
            "width": 390, "height": 1000, "deviceScaleFactor": 1, "mobile": True,
        })
        browser.wait("window.innerWidth === 390")
        browser.evaluate("window.scrollTo(0,0)")
        overflow = browser.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        assert not overflow, "Horizontal overflow at phone width"
        browser.screenshot(output_dir / "mobile.png")
        if did_predict:
            browser.evaluate("document.querySelector('#result-panel').scrollIntoView({block:'start',behavior:'instant'})")
            browser.screenshot(output_dir / "mobile-result.png")
        browser.command("Emulation.setDeviceMetricsOverride", {
            "width": 360, "height": 1000, "deviceScaleFactor": 1, "mobile": True,
        })
        browser.wait("window.innerWidth === 360")
        assert not browser.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "Horizontal overflow at360px"
        browser.evaluate("document.querySelector('[data-view=danh-muc]').click()")
        browser.wait("!document.querySelector('#view-danh-muc').hidden")
        assert not browser.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "Catalog page overflow"
        clipboard = None
        if has_sample or uploaded:
            browser.evaluate("document.querySelector('[data-view=nhan-dien]').click()")
            browser.wait("!document.querySelector('#view-nhan-dien').hidden")
            clipboard = clipboard_smoke(browser)
            browser.screenshot(output_dir / "clipboard-paste.png")
        assert not browser.console_errors, browser.console_errors
        return {"browser": executable.name, "model_id": model_id, "prediction_interaction": did_predict,
                "clipboard": clipboard,
                "file_upload_interaction": uploaded, "upload_roi": upload_roi if uploaded else None,
                "catalog_search": "passed", "removed_experiment_and_download": "passed", "theme_toggle": "passed",
                "mobile_overflow": overflow, "console_errors": browser.console_errors,
                "screenshots": str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir)}
    finally:
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--http-only", action="store_true")
    parser.add_argument("--browser")
    parser.add_argument("--upload-image", type=Path, help="Optional local PNG/JPEG/WebP to drive the browser file picker")
    parser.add_argument("--upload-roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Optional inclusive ROI for the uploaded image")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "web" / "smoke")
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Smoke tests only target a local HTTP server")
    result = {"http": http_smoke(args.url)}
    if not args.http_only:
        result["browser"] = browser_smoke(args.url, args.output_dir, args.browser, args.upload_image, args.upload_roi)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
