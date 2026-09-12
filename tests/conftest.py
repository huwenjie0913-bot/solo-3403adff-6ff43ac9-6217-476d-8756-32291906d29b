import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="balancing_test_")
os.environ["BALANCING_DATABASE_URL"] = f"sqlite:///{_tmpdir}/test.db"
