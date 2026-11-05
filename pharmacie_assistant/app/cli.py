"""CLI :  python -m app.cli "wach kayn doliprane 1000 ?"  [--mode rules|ml|hybrid] [--no-fulfill]"""
import argparse
import json

from .pipeline import Assistant

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--mode", default=None)
    p.add_argument("--no-fulfill", action="store_true")
    a = p.parse_args()
    kw = {"intent_mode": a.mode} if a.mode else {}
    print(json.dumps(Assistant(**kw).analyze(a.text, do_fulfill=not a.no_fulfill), ensure_ascii=False, indent=2))
