"""To-scale sketch of the mirror pre-demonstrator with the aerogel tile (arc_xe_demonstrator_sketch.png)."""
from xerich import cli
from xerich.plotting import save
from xerich.sketch import draw


def main():
    out = cli.outdir(cli.parser(__doc__).parse_args())
    save(out, 'arc_xe_demonstrator_sketch.png', dpi=150, fig=draw(aerogel=True))


if __name__ == '__main__':
    main()
