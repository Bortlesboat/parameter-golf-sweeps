from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class RunSweepsTests(unittest.TestCase):
    def test_writes_ranked_summary_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "run_sweeps.py"

        with tempfile.TemporaryDirectory() as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            workdir = tmp_dir / "workspace"
            workdir.mkdir()

            fake_train = workdir / "fake_train.py"
            fake_train.write_text(
                textwrap.dedent(
                    """
                    import os

                    print("Code size: 47642 bytes")
                    print("Total submission size int8+zlib: 15863489 bytes")
                    print(
                        "final_int8_zlib_roundtrip_exact val_loss:2.07269931 "
                        f"val_bpb:{os.environ['FAKE_VAL_BPB']}"
                    )
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config_path = tmp_dir / "config.json"
            output_dir = tmp_dir / "outputs"
            config_path.write_text(
                json.dumps(
                    {
                        "workdir": str(workdir),
                        "log_dir": str(output_dir),
                        "command": [sys.executable, "fake_train.py"],
                        "runs": [
                            {"name": "worse", "env": {"FAKE_VAL_BPB": "1.30"}},
                            {"name": "better", "env": {"FAKE_VAL_BPB": "1.10"}},
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(  # noqa: S603
                [sys.executable, str(script_path), "--config", str(config_path)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)

            summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertIn("| better | 0 | 2.07269931 | 1.10000000 | 47642 | 15863489 | better.log |", summary_md)
            self.assertIn("| worse | 0 | 2.07269931 | 1.30000000 | 47642 | 15863489 | worse.log |", summary_md)
            self.assertLess(summary_md.index("| better |"), summary_md.index("| worse |"))


if __name__ == "__main__":
    unittest.main()
