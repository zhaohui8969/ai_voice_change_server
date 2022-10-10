import argparse


def config(cli_lines=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=7780)
    parser.add_argument('--model_dir', default="/code/pt_model")
    args = parser.parse_args(args=cli_lines)

    print(args)
    return args
