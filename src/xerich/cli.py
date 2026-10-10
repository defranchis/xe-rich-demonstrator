"""Command-line helpers shared by the study drivers."""
import argparse
import os
from pathlib import Path


def default_outdir():
    """$XERICH_OUT if set, else xerich-output/ next to the repository checkout."""
    env = os.environ.get('XERICH_OUT')
    return Path(env) if env else Path(__file__).resolve().parents[3]/'xerich-output'


def parser(doc):
    """Argument parser with the common --outdir option."""
    p = argparse.ArgumentParser(description=doc)
    p.add_argument('--outdir', type=Path, default=None,
                   help='output directory (default: $XERICH_OUT, else xerich-output/ next to the repository)')
    return p


def outdir(args):
    """Resolve and create the output directory of parsed arguments."""
    d = args.outdir if args.outdir is not None else default_outdir()
    d.mkdir(parents=True, exist_ok=True)
    return d
