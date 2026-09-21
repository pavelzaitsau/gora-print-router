# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for CLI-level concerns in src.core.router:

* `_reject_duplicate_args` — fail-fast when the user passes the same option
  twice. argparse would silently keep only the last value, which is the kind
  of bug that bit us with launch.json (two preset rows both un-commented).
"""

from __future__ import annotations

import argparse

import pytest
from src.core.router import _reject_duplicate_args


def _make_parser() -> argparse.ArgumentParser:
    """Minimal parser mirroring the flags relevant to the duplicate-check."""
    p = argparse.ArgumentParser()
    p.add_argument("track_gpx", nargs="?")
    p.add_argument("--model-size-x", type=float)
    p.add_argument("--model-size-y", type=float)
    p.add_argument("--model-shape", choices=["rectangle", "hexagon", "circle"])
    p.add_argument("--auto-rotate", action="store_true")
    p.add_argument("-o", "--output", type=str)
    return p


class TestRejectDuplicateArgs:
    def test_unique_args_ok(self):
        argv = [
            "track.gpx",
            "--model-size-x",
            "90",
            "--model-size-y",
            "70",
            "--auto-rotate",
        ]
        # No raise.
        _reject_duplicate_args(argv, _make_parser())

    def test_duplicate_value_flag_raises(self):
        argv = ["--model-size-x", "90", "--model-size-x", "100"]
        with pytest.raises(SystemExit):
            _reject_duplicate_args(argv, _make_parser())

    def test_duplicate_bool_flag_raises(self):
        argv = ["--auto-rotate", "--auto-rotate"]
        with pytest.raises(SystemExit):
            _reject_duplicate_args(argv, _make_parser())

    def test_equals_form_counted(self):
        # `--model-size-x=90` is a single token but counts the same as the
        # whitespace form.
        argv = ["--model-size-x=90", "--model-size-x", "100"]
        with pytest.raises(SystemExit):
            _reject_duplicate_args(argv, _make_parser())

    def test_short_form_duplicate(self):
        argv = ["-o", "a", "-o", "b"]
        with pytest.raises(SystemExit):
            _reject_duplicate_args(argv, _make_parser())

    def test_short_and_long_same_option_duplicate(self):
        # `-o` and `--output` are aliases for the same action. Currently the
        # check is by literal token, so a short+long combo passes — document
        # that with this xfail-style test: if someone tightens the check
        # later, this is the spec they should target.
        argv = ["-o", "a", "--output", "b"]
        # Today: no raise (per-token counting).
        _reject_duplicate_args(argv, _make_parser())

    def test_lists_all_duplicates(self, capsys):
        argv = [
            "--model-size-x",
            "1",
            "--model-size-x",
            "2",
            "--model-size-y",
            "3",
            "--model-size-y",
            "4",
        ]
        with pytest.raises(SystemExit):
            _reject_duplicate_args(argv, _make_parser())
        err = capsys.readouterr().err
        assert "--model-size-x" in err
        assert "--model-size-y" in err

    def test_unknown_flag_ignored(self):
        # Tokens that aren't registered options shouldn't trigger the check
        # — argparse itself will fail on them later with a clear message.
        argv = ["--definitely-not-a-flag", "x", "--definitely-not-a-flag", "y"]
        _reject_duplicate_args(argv, _make_parser())

    def test_positional_repeats_ignored(self):
        # Two raw strings that look like tracks but argparse only allows one.
        # The duplicate check shouldn't fire for positionals.
        argv = ["track1.gpx", "track2.gpx"]
        _reject_duplicate_args(argv, _make_parser())


class TestFailuresReachTheUser:
    """A command-line tool reports a bad input; it does not print a stack trace."""

    def test_a_missing_gpx_is_an_error_message_not_a_traceback(self, monkeypatch, capsys):
        """Every failure in the pipeline currently escapes `main` uncaught.

        The messages themselves are good. What reaches the user is a Python
        traceback with `src/core/...` frames above them, which reads as a crash
        in the tool rather than a problem with the input.
        """
        import src.core.router as router

        monkeypatch.setattr("sys.argv", ["main.py", "definitely-not-here.gpx"])
        with pytest.raises(SystemExit) as exit_info:
            router.main()

        assert exit_info.value.code == 2, "a bad input should exit 2, the argparse convention"
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "definitely-not-here.gpx" in err


class TestConsoleEncoding:
    """Progress output must not kill a run it only meant to narrate."""

    def test_a_narrow_console_does_not_stop_the_run(self):
        """Windows redirects stdout through the ANSI code page, not UTF-8.

        The run prints check marks, warning signs and arrows. Encoding one of
        those to cp1252 raises, so a redirected run died partway through a
        multi-minute job with `UnicodeEncodeError` and wrote no model. Losing
        the glyph is cheap; losing the model is not.
        """
        import io
        import sys

        import src.core.router as router

        # The premise: this is the code page a redirected Windows console uses,
        # and it has no check mark.
        with pytest.raises(UnicodeEncodeError):
            "\u2713".encode("cp1252")

        raw = io.BytesIO()
        guarded = io.TextIOWrapper(raw, encoding="cp1252", newline="")
        original = sys.stdout
        sys.stdout = guarded
        try:
            router._survive_a_narrow_console()
            print("\u2713 guarded")
            guarded.flush()
        finally:
            sys.stdout = original

        written = raw.getvalue()
        assert b"guarded" in written, "the line never reached the stream"
        assert b"\xe2\x9c\x93" not in written, "the glyph should have been replaced, not encoded"


def router_module():
    import src.core.router as router

    return router
