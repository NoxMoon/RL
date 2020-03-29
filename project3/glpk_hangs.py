import cvxopt
from cvxopt.solvers import lp
#from cvxopt.solvers import conelp as lp

from cvxopt import matrix, glpk
import pickle
import sys

print('python version:', sys.version)
print('cvxopt version:', cvxopt.__version__)

#glpk.options['msg_lev'] = 'GLP_MSG_OFF'

args = pickle.load(open('glpk_hangs.pkl', 'rb'))

ret = lp(matrix(args['c']), matrix(args['G']), matrix(args['h']), solver=args['solver'])
print(ret)
