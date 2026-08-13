# -*- coding: utf-8 -*-
import aiohttp
import asyncio

API_BASE = "https://api.phira.cn"
CHART_API = f"{API_BASE}/chart/{{chart_id}}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://phira.cn/"
}

async def test_api():
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout, headers=HEADERS) as session:
        # 测试几个 ID
        for chart_id in [1, 100, 1000, 2000, 5000]:
            url = CHART_API.format(chart_id=chart_id)
            try:
                async with session.get(url) as resp:
                    print(f"ID {chart_id}: status={resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"  code: {data.get('code')}")
                        print(f"  name: {data.get('name', 'N/A')}")
                        print(f"  file: {data.get('file', 'N/A')[:50]}...")
                    else:
                        text = await resp.text()
                        print(f"  response: {text[:100]}")
            except Exception as e:
                print(f"ID {chart_id}: error - {e}")

asyncio.run(test_api())
