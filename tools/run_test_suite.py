"""Run meaningful unit checks and save a machine-readable outcome."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py',top_level_dir=str(ROOT))
result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            skipped=len(result.skipped),passed=result.wasSuccessful())
(ROOT/'results/unit_test_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
sys.exit(0 if result.wasSuccessful() else 1)
