"""pta.worker with the fake models incl. the fake SAM3: python tests/fake_worker_people.py job.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fakes_people  # noqa: installs the fake models
from pta import worker
sys.exit(worker.main())
