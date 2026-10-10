import os
import sys

# Permite `pytest` a partir de project/ ou da raiz do repositório
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
