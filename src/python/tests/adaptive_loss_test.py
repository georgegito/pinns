import os
from dotenv import load_dotenv
import sys

load_dotenv()

lib_dir = os.environ.get("LIB_DIR")
sys.path.append(lib_dir)

import utils

relobralo = utils.ReLoBRaLo()

l2 = [1]
l3 = [1]
l1 = [1]

L = [l1, l2, l3]

lambdas = relobralo.compute_next_lambdas(L)

print(lambdas)