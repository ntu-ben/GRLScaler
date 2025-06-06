#!/usr/bin/env python3
import shutil
import re
from pathlib import Path
import argparse

def organize_hpa(directory: Path):
    # 只處理目錄下的 YAML 定義檔
    for file in directory.glob('*.yaml'):
        name = file.name
        # 抓取 CPU 和 MEM 閾值
        m_cpu = re.search(r'cpu-(\d+)', name)
        m_mem = re.search(r'mem-(\d+)', name)

        if m_cpu and m_mem:
            group = f"cpu-{m_cpu.group(1)}-mem-{m_mem.group(1)}"
        elif m_cpu:
            group = f"cpu-{m_cpu.group(1)}"
        elif m_mem:
            group = f"mem-{m_mem.group(1)}"
        else:
            group = "others"

        dest = directory / group
        dest.mkdir(exist_ok=True)
        shutil.move(str(file), str(dest / file.name))

    # 列印結果摘要
    print("✅ 已依閾值分類完畢：")
    for d in sorted(directory.iterdir()):
        if d.is_dir():
            count = len(list(d.glob('*.yaml')))
            print(f"  {d.name:20} ({count} 個檔案)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize HPA YAML by thresholds')
    parser.add_argument('target_dir', nargs='?', default='.', help='目標目錄，預設當前目錄')
    args = parser.parse_args()

    base_dir = Path(args.target_dir).resolve()
    organize_hpa(base_dir)

