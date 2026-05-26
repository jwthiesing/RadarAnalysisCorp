"""Application entry point.

Uses ``qasync`` to run a single event loop that serves both Qt's GUI events
and ``asyncio`` (needed by ``aiortc`` / ``aiohttp`` for multiplayer).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Startup housekeeping: purge cache files older than ~30 days. Cheap.
    from radar_warning_game.data.cache import purge_cache_older_than
    purge_cache_older_than()

    from PyQt6.QtWidgets import QApplication
    import qasync

    from radar_warning_game.ui.app import MainWindow

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow(local_player_name="You")
    window.show()

    # Graceful Ctrl-C inside the qasync loop
    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
