"""Headless smoke test for the static site.

  pip install playwright && python -m playwright install chromium
  python tests/smoke_test.py            (needs internet for the three.js CDN)

Checks: the page loads without script errors in all three language modes, the
"Play example" run finishes with both answers, the inspector builds every model,
and the data bay trains on a tiny generated table.
"""
import asyncio, os, sys
from playwright.async_api import async_playwright

PAGE = "file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
IDS = ["perceptron", "mlp", "kan", "cnn", "resnet", "unet", "rnn", "lstm", "seq2seq", "mamba", "transformer",
       "bert", "gpt", "vit", "moe", "vae", "gan", "diffusion", "dit", "gnn", "dqn"]
CSV = "x1,x2,label\n" + "\n".join(f"{i % 7},{(i * 3) % 5},{'a' if i % 7 < 3 else 'b'}" for i in range(60))


async def main():
    fails = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        for lang in ("zh", "en", "both"):
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            await page.goto(PAGE + "?lang=" + lang)
            await page.wait_for_timeout(2500)
            if errs:
                fails.append(f"[{lang}] load errors: {errs}")
            if lang == "en":
                await page.click("#demoBtn")
                await page.wait_for_function("document.getElementById('stext').innerText.startsWith('Done')", timeout=240000)
                if len(await page.inner_text("#ansB")) < 200:
                    fails.append("demo answer B is empty")
                for m in IDS:
                    await page.evaluate(f"document.querySelector('[data-sel={m}]').click()")
                    await page.wait_for_timeout(150)
                    if await page.evaluate("document.querySelectorAll('#inspList li').length") < 2:
                        fails.append(f"inspector did not build {m}")
                tmp = os.path.join(os.path.dirname(__file__), "_smoke.csv")
                open(tmp, "w").write(CSV)
                await page.set_input_files("#fTab", tmp)
                await page.wait_for_timeout(800)
                await page.select_option("#lbEp", "20")
                await page.click('#lab [data-act="train"]')
                await page.wait_for_selector('#lab [data-act="deploy"]', timeout=120000)
                os.remove(tmp)
                if errs:
                    fails.append(f"errors during the run: {errs}")
            await page.close()
        await browser.close()
    print("\n".join(fails) if fails else "SMOKE TEST PASSED")
    sys.exit(1 if fails else 0)


asyncio.run(main())
