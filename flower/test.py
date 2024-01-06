import sys
sys.path.append('..')
from utils.config_parser import Parser

args = Parser().parse()
print('hello')