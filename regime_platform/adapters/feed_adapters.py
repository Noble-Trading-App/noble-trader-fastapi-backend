"""
Real-Time OHLCV Feed Adapters
═══════════════════════════════

Async adapters that stream live OHLCV bars into `StreamSession.tick()`.

Supported brokers
─────────────────
  • Alpaca        — REST + WebSocket (crypto + equities)
  • Binance       — WebSocket kline stream (crypto)
  • Interactive Brokers — TWS API via ib_async (equities, futures, FX)

Architecture
────────────
Each adapter is an async context manager that yields OHLCV bars.
A shared `FeedManager` runs adapters as asyncio tasks and routes
bars into the session registry.

Usage
─────
  # Alpaca
  async with AlpacaFeedAdapter(symbols=["SPY","QQQ"], bar_size="1Min") as feed:
      async for bar in feed:
          await session.tick(bar.close, bar.ts)

  # FeedManager (recommended — handles reconnect + multi-symbol routing)
  mgr = FeedManager()
  await mgr.add(AlpacaFeedAdapter(symbols=["SPY"], bar_size="1Min"))
  await mgr.add(BinanceFeedAdapter(symbols=["BTCUSDT"], interval="1m"))
  await mgr.start()   # blocks; ctrl+c to stop

Environment variables required per adapter
──────────────────────────────────────────
  Alpaca:   ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
            (paper: https://paper-api.alpaca.markets)
  Binance:  BINANCE_API_KEY, BINANCE_SECRET_KEY (read-only key sufficient)
  IB:       IB_HOST (default 127.0.0.1), IB_PORT (default 7497), IB_CLIENT_ID

Install requirements (not in base requirements.txt — optional)
──────────────────────────────────────────────────────────────
  pip install alpaca-py websockets aiohttp ib_async
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

log = logging.getLogger("regime.adapters")


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class OHLCVBar:
    """Standardised OHLCV bar from any feed adapter."""
    symbol:    str
    ts:        float      # unix timestamp (bar close)
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    source:    str = ""   # "alpaca" | "binance" | "ib"


# ─── Abstract base ────────────────────────────────────────────────────────────

class FeedAdapter(ABC):
    """
    Async OHLCV feed adapter.

    Subclass and implement `stream()` — an async generator that yields `OHLCVBar`.
    The base class provides reconnect logic with exponential backoff.
    """

    MAX_RECONNECT_ATTEMPTS = 10
    BASE_BACKOFF_SECS      = 1.0

    def __init__(self, symbols: list[str], reconnect: bool = True):
        self.symbols   = [s.upper() for s in symbols]
        self.reconnect = reconnect
        self._running  = False
        self._bar_count: dict[str, int] = {s: 0 for s in self.symbols}

    @abstractmethod
    async def stream(self) -> AsyncGenerator[OHLCVBar, None]:
        """Yield live OHLCVBars. Raise to trigger reconnect."""
        ...

    async def __aenter__(self) -> "FeedAdapter":
        self._running = True
        return self

    async def __aexit__(self, *args) -> None:
        self._running = False

    async def run_with_reconnect(self) -> AsyncGenerator[OHLCVBar, None]:
        """Wrapper that retries `stream()` on exception with exponential backoff."""
        attempts = 0
        while self._running:
            try:
                attempts = 0
                async for bar in self.stream():
                    self._bar_count[bar.symbol] = self._bar_count.get(bar.symbol, 0) + 1
                    yield bar
            except Exception as exc:
                attempts += 1
                if attempts > self.MAX_RECONNECT_ATTEMPTS:
                    log.error(f"{self.__class__.__name__}: max reconnect attempts reached. Stopping.")
                    return
                wait = self.BASE_BACKOFF_SECS * (2 ** min(attempts, 8))
                log.warning(f"{self.__class__.__name__}: disconnected ({exc}). Reconnecting in {wait:.1f}s (attempt {attempts})")
                await asyncio.sleep(wait)

    @property
    def stats(self) -> dict:
        return {"adapter": self.__class__.__name__, "symbols": self.symbols, "bars": self._bar_count}


# ─── Alpaca ───────────────────────────────────────────────────────────────────

class AlpacaFeedAdapter(FeedAdapter):
    """
    Alpaca Markets real-time bar feed via alpaca-py WebSocket client.

    Streams 1-minute (or configurable) bars for equities and crypto.

    Required env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    Install: pip install alpaca-py

    Paper trading URL: https://paper-api.alpaca.markets
    Live trading URL:  https://api.alpaca.markets
    """

    def __init__(
        self,
        symbols:  list[str],
        bar_size: str = "1Min",
        feed:     str = "sip",           # "sip" (US equities) or "iex"
        reconnect: bool = True,
    ):
        super().__init__(symbols, reconnect)
        self.bar_size = bar_size
        self.feed     = feed
        self._api_key    = os.getenv("ALPACA_API_KEY", "")
        self._secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self._base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    async def stream(self) -> AsyncGenerator[OHLCVBar, None]:
        """Stream bars from Alpaca WebSocket data stream."""
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.enums import DataFeed
        except ImportError:
            raise RuntimeError("alpaca-py not installed: pip install alpaca-py")

        if not self._api_key or not self._secret_key:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY env vars required")

        queue: asyncio.Queue[OHLCVBar] = asyncio.Queue()

        wss = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=DataFeed.SIP if self.feed == "sip" else DataFeed.IEX,
        )

        async def _bar_handler(bar):
            await queue.put(OHLCVBar(
                symbol=bar.symbol,
                ts=bar.timestamp.timestamp(),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                source="alpaca",
            ))

        for sym in self.symbols:
            wss.subscribe_bars(_bar_handler, sym)

        # Run WebSocket in background task
        ws_task = asyncio.create_task(wss.run())
        try:
            while self._running:
                try:
                    bar = await asyncio.wait_for(queue.get(), timeout=60.0)
                    yield bar
                except asyncio.TimeoutError:
                    log.debug("AlpacaFeedAdapter: heartbeat timeout, still connected")
        finally:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass


# ─── Binance ──────────────────────────────────────────────────────────────────

class BinanceFeedAdapter(FeedAdapter):
    """
    Binance kline (candlestick) WebSocket stream.

    Streams OHLCV bars at the specified interval for any Binance spot pair.

    Required env vars: None (public stream — no auth needed for klines)
    Optional: BINANCE_TESTNET=true for testnet
    Install: pip install websockets

    Intervals: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
    """

    MAINNET_WS = "wss://stream.binance.com:9443/stream"
    TESTNET_WS = "wss://testnet.binance.vision/stream"

    def __init__(
        self,
        symbols:  list[str],          # e.g. ["BTCUSDT", "ETHUSDT"]
        interval: str = "1m",
        reconnect: bool = True,
    ):
        super().__init__(symbols, reconnect)
        self.interval = interval
        self._testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

    async def stream(self) -> AsyncGenerator[OHLCVBar, None]:
        """Stream kline bars from Binance WebSocket."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets not installed: pip install websockets")

        # Build combined stream URL
        streams = "/".join(
            f"{sym.lower()}@kline_{self.interval}" for sym in self.symbols
        )
        ws_url = f"{self.TESTNET_WS if self._testnet else self.MAINNET_WS}?streams={streams}"
        log.info(f"BinanceFeedAdapter connecting: {ws_url}")

        async with websockets.connect(ws_url, ping_interval=30) as ws:
            async for raw in ws:
                if not self._running:
                    break
                data = json.loads(raw)
                k    = data.get("data", {}).get("k", {})

                # Only yield on closed candle
                if not k.get("x", False):
                    continue

                symbol = k.get("s", "UNKNOWN")
                yield OHLCVBar(
                    symbol=symbol,
                    ts=float(k["T"]) / 1000.0,   # close timestamp ms → s
                    open=float(k["o"]),
                    high=float(k["h"]),
                    low=float(k["l"]),
                    close=float(k["c"]),
                    volume=float(k["v"]),
                    source="binance",
                )


# ─── Interactive Brokers ──────────────────────────────────────────────────────

class IBFeedAdapter(FeedAdapter):
    """
    Interactive Brokers real-time bar feed via ib_async (asyncio TWS API).

    Streams 5-second or 1-minute real-time bars for equities, futures, FX.

    Required env vars: IB_HOST, IB_PORT, IB_CLIENT_ID
    Install: pip install ib_async
    Prerequisite: TWS or IB Gateway must be running with API enabled.

    IB_PORT:  7497 (TWS paper), 7496 (TWS live), 4002 (Gateway paper), 4001 (live)
    """

    def __init__(
        self,
        symbols:   list[str],
        bar_size:  str = "5 secs",    # IB bar size string
        what_to_show: str = "TRADES",
        sec_type:  str = "STK",
        exchange:  str = "SMART",
        currency:  str = "USD",
        reconnect: bool = True,
    ):
        super().__init__(symbols, reconnect)
        self.bar_size     = bar_size
        self.what_to_show = what_to_show
        self.sec_type     = sec_type
        self.exchange     = exchange
        self.currency     = currency
        self._host        = os.getenv("IB_HOST",      "127.0.0.1")
        self._port        = int(os.getenv("IB_PORT",  "7497"))
        self._client_id   = int(os.getenv("IB_CLIENT_ID", "1"))

    async def stream(self) -> AsyncGenerator[OHLCVBar, None]:
        """Stream real-time bars from IB TWS."""
        try:
            from ib_async import IB, Stock, RealTimeBarList
        except ImportError:
            raise RuntimeError("ib_async not installed: pip install ib_async")

        ib = IB()
        await ib.connectAsync(self._host, self._port, clientId=self._client_id)
        log.info(f"IBFeedAdapter connected: {self._host}:{self._port} (client={self._client_id})")

        queue: asyncio.Queue[OHLCVBar] = asyncio.Queue()

        contracts = [
            Stock(sym, self.exchange, self.currency)
            for sym in self.symbols
        ]
        qualified = await ib.qualifyContractsAsync(*contracts)

        bar_lists: list[RealTimeBarList] = []
        for contract in qualified:
            bars = ib.reqRealTimeBars(
                contract,
                barSize=5,   # IB only supports 5s for real-time bars
                whatToShow=self.what_to_show,
                useRTH=True,
            )
            bars.updateEvent += lambda bar_list, changed: asyncio.ensure_future(
                queue.put(OHLCVBar(
                    symbol=bar_list.contract.symbol,
                    ts=float(bar_list[-1].time),
                    open=float(bar_list[-1].open),
                    high=float(bar_list[-1].high),
                    low=float(bar_list[-1].low),
                    close=float(bar_list[-1].close),
                    volume=float(bar_list[-1].volume),
                    source="ib",
                ))
            )
            bar_lists.append(bars)

        try:
            while self._running and ib.isConnected():
                try:
                    bar = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield bar
                except asyncio.TimeoutError:
                    log.debug("IBFeedAdapter: heartbeat")
        finally:
            for bars in bar_lists:
                ib.cancelRealTimeBars(bars)
            ib.disconnect()
            log.info("IBFeedAdapter disconnected")


# ─── Feed Manager ─────────────────────────────────────────────────────────────

class FeedManager:
    """
    Manages multiple FeedAdapters and routes bars into the SessionRegistry.

    Usage
    ─────
    mgr = FeedManager()
    mgr.add(AlpacaFeedAdapter(["SPY","QQQ"], bar_size="1Min"))
    mgr.add(BinanceFeedAdapter(["BTCUSDT"], interval="1m"))
    await mgr.start()   # blocks until stopped

    Each bar automatically calls:
      1. registry.get(symbol)
      2. session.tick(bar.close, bar.ts)
    """

    def __init__(self):
        self._adapters: list[FeedAdapter] = []
        self._tasks:    list[asyncio.Task] = []
        self._running = False
        self._bar_count: dict[str, int] = {}

    def add(self, adapter: FeedAdapter) -> "FeedManager":
        self._adapters.append(adapter)
        return self

    async def start(self) -> None:
        """Start all adapters. Blocks until stop() is called."""
        from ..services.registry import registry

        self._running = True
        log.info(f"FeedManager starting {len(self._adapters)} adapter(s)")

        async def _run_adapter(adapter: FeedAdapter):
            async with adapter:
                async for bar in adapter.run_with_reconnect():
                    if not self._running:
                        break
                    session = await registry.get(bar.symbol)
                    if session is not None:
                        await session.tick(bar.close, bar.ts)
                        self._bar_count[bar.symbol] = self._bar_count.get(bar.symbol, 0) + 1
                    else:
                        log.debug(f"FeedManager: no session for {bar.symbol} — bar dropped (seed first)")

        self._tasks = [
            asyncio.create_task(_run_adapter(a))
            for a in self._adapters
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Gracefully stop all adapter tasks."""
        self._running = False
        for a in self._adapters:
            a._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("FeedManager stopped")

    @property
    def stats(self) -> dict:
        return {
            "adapters": [a.stats for a in self._adapters],
            "total_bars": self._bar_count,
        }
