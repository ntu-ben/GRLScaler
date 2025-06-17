import asyncio
import httpx
from queue import Queue


async def fetch_json(client: httpx.AsyncClient, url: str):
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()


async def pull_loop(queue: Queue, edges_url: str, metrics_url: str, interval: int = 30):
    """Periodically fetch Linkerd edges and Prometheus metrics."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                edges = await fetch_json(client, edges_url)
                metrics = await fetch_json(client, metrics_url)
                queue.put({'edges': edges, 'metrics': metrics})
            except Exception as exc:
                queue.put({'error': str(exc)})
            await asyncio.sleep(interval)


async def _run_cli(edges_url: str, metrics_url: str, interval: int):
    q = Queue()
    task = asyncio.create_task(pull_loop(q, edges_url, metrics_url, interval))
    try:
        while True:
            item = q.get()
            print(item)
    except KeyboardInterrupt:
        task.cancel()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Linkerd/Prometheus puller")
    parser.add_argument("--edges-url", required=True, help="Linkerd edges API")
    parser.add_argument("--metrics-url", required=True, help="Prometheus metrics API")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    asyncio.run(_run_cli(args.edges_url, args.metrics_url, args.interval))


if __name__ == "__main__":
    main()
