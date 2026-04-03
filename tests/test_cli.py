from kaggle_nvida.cli import build_parser


def test_cli_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "kaggle-nvida"
