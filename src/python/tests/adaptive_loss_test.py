import os
from dotenv import load_dotenv
import sys

load_dotenv()

lib_dir = os.environ.get("LIB_DIR")
sys.path.append(lib_dir)

import utils

relobralo = utils.ReLoBRaLo()

l2 = [1.02, 2.424, 3.232, 4.]
l3 = [1.011, 2, 3.42, 4.33]
l1 = [1.032, 2.444, 3.1222, 4.2]

L = [l1, l2, l3]

lambdas = relobralo.compute_next_lambdas(L)

print(lambdas)