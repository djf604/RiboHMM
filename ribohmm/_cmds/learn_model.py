from ribohmm._cmds._main import execute_ribohmm, common_args, learn_args


def populate_parser(parser):
    common_args(parser)
    learn_args(parser)


def main(args=None):
    execute_ribohmm(args, learn=True, infer=False)


if __name__ == '__main__':
    main()
