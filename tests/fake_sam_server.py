"""laa.sam_server with the fake SAM3 (live clicks): python tests/fake_sam_server.py job.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fakes_people  # noqa: installs the fake models
from laa import sam_server
sys.exit(sam_server.main())
