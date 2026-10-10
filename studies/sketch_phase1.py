"""To-scale sketch of configuration 1, the Xe gas ring without aerogel (arc_xe_demonstrator_phase1_gas_only.png)."""
from xerich import cli
from xerich.plotting import save
from xerich.sketch import draw


def main():
    out = cli.outdir(cli.parser(__doc__).parse_args())
    save(out, 'arc_xe_demonstrator_phase1_gas_only.png', dpi=150, fig=draw(aerogel=False))


if __name__ == '__main__':
    main()
