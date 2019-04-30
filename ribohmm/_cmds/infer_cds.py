from ribohmm._cmds._main import execute_ribohmm, common_args, infer_args


def populate_parser(parser):
    common_args(parser)
    infer_args(parser)


def main(args=None):
    execute_ribohmm(args, learn=False, infer=True)


if __name__ == '__main__':
    main()
