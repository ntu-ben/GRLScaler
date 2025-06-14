import argparse
import time
from . import dataloader, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ns', required=True)
    p.add_argument('--step', type=int, default=30)
    p.add_argument('--prom-url', default=settings.PROM_URL)
    p.add_argument('--viz-url', default=settings.VIZ_URL)
    p.add_argument('--once', action='store_true', help='run only one iteration')
    args = p.parse_args()

    while True:
        edges = dataloader.fetch_edges(args.ns, viz_url=args.viz_url, prom_url=args.prom_url)
        nodes = dataloader.fetch_node_features(args.ns, prom_url=args.prom_url)
        dataloader.save_data(edges, nodes)
        print(f"edges  E={len(edges)}, nodes N={len(nodes)}")
        if args.once:
            break
        time.sleep(args.step)


if __name__ == '__main__':
    main()
