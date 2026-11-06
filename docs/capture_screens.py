import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = "http://localhost:8501/"
OUT = Path(__file__).resolve().parent / "screens"
PAGES = {
    "patient_pima": "?dataset=pima&rules=4&patient=0",
    "patient_early": "?dataset=early&rules=4&patient=3",
    "patient_cdc": "?dataset=cdc&rules=8&patient=12",
    "patient_variant": "?dataset=early&variant=gauss-0&patient=3",
    "results": "?tab=results",
    "data": "?tab=data",
}
PORT = 9333


async def call(ws, method, params=None, _id=[0]):
    _id[0] += 1
    await ws.send(json.dumps({"id": _id[0], "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == _id[0]:
            return msg.get("result", {})


async def main():
    OUT.mkdir(exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--remote-debugging-port={PORT}", "--window-size=1500,1000",
                             f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"])
    try:
        for _ in range(40):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except Exception:
                time.sleep(0.5)
        page = next(t for t in targets if t["type"] == "page")
        ws = await websockets.connect(page["webSocketDebuggerUrl"], max_size=2 ** 28)
        await call(ws, "Page.enable")
        for name, query in PAGES.items():
            await call(ws, "Emulation.setDeviceMetricsOverride",
                       {"width": 1500, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
            await call(ws, "Page.navigate", {"url": BASE + query})
            for _ in range(120):
                await asyncio.sleep(1)
                r = await call(ws, "Runtime.evaluate", {"returnByValue": True, "expression":
                    "!!document.querySelector('[data-testid=\"stApp\"]') && "
                    "document.querySelector('[data-testid=\"stApp\"]').getAttribute('data-test-script-state')"})
                if r.get("result", {}).get("value") == "notRunning":
                    break
            await asyncio.sleep(4)
            r = await call(ws, "Runtime.evaluate", {"returnByValue": True, "expression":
                "Math.max(...[...document.querySelectorAll('*')].map(e => e.scrollHeight))"})
            height = min(int(r["result"]["value"]) + 40, 9000)
            await call(ws, "Emulation.setDeviceMetricsOverride",
                       {"width": 1500, "height": height, "deviceScaleFactor": 1, "mobile": False})
            await asyncio.sleep(3)
            shot = await call(ws, "Page.captureScreenshot", {"format": "png"})
            (OUT / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
            print("saved", name, height)
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
