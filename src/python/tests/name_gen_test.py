import os
import sys
from dotenv import load_dotenv

load_dotenv()
lib_dir = os.environ.get("LOCAL_LIB_DIR")
sys.path.append(lib_dir)

from utils import NameGenerator

nameGenerator = NameGenerator()
name = nameGenerator.generate_name()

print(name)
