"""pta.worker with the fake models: python tests/fake_worker.py job.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fakes  # noqa: installs the fake models
from pta import worker
sys.exit(worker.main())
