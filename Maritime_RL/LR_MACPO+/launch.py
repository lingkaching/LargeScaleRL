"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 26 Apr 2019
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import os
from parameters_11 import AGENT, LOAD_MODEL
from optparse import OptionParser

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

def main():
    parser = OptionParser()
    parser.add_option("-r", "--runID", type='int', dest='runID', default=0) 
    (options, args) = parser.parse_args()  

    para_file = open("parameters.py", "r")
    list_of_lines = para_file.readlines()
    list_of_lines[0] = "SEED = " + str(options.runID) +  "\n"

    para_file = open("parameters.py", "w")
    para_file.writelines(list_of_lines)
    para_file.close()

    if LOAD_MODEL:
        if AGENT == "pg_fict_dcp":
            os.system("python eval_load_baseline.py -a " + AGENT)
        elif AGENT == "tmin":
            os.system("python eval_load_baseline.py -a " + AGENT)
        else:
            print ("Error: Agent not recognized !")
    else:
        if AGENT == "pg_fict_dcp":
            os.system("python main_baseline.py -a " + AGENT + " -r " +str(options.runID))
        elif AGENT == "tmin":
            os.system("python main_baseline.py -a " + AGENT + " -r " +str(options.runID))
        else:
            print ("Error: Agent not recognized !")


# =============================================================================== #

if __name__ == '__main__':
    main()
    