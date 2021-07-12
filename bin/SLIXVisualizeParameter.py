import os
import re
import sys
sys.path.append(os.getcwd())
sys.path.append("..")

from SLIX._cmd import main_visualize

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main_visualize())
