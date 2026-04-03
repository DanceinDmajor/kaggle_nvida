import unittest

from kaggle_nvida.cli import build_parser


class CliTests(unittest.TestCase):
    def test_cli_parser_builds(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.prog, "kaggle-nvida")
