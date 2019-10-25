from typing import List, Callable
import exact_real_program
import os
import numpy as np
import subprocess
from tabulate import tabulate
from datetime import timedelta
from timeit import default_timer as timer


def time_wrap(f, args):
    """Apply the arguments to the function and return the time and result. """
    start_time = timer()
    res = f(*args)
    end_time = timer()
    return timedelta(seconds=end_time - start_time), res

def cast_input(to_cast):
    return exact_real_program.ExactConstant(to_cast) if isinstance(to_cast, (int, float)) else to_cast


def create_pdf_from_table(filename: str, table_info: np.array, headers: List):
    table = tabulate(table_info, tablefmt="latex", headers=headers)
    # print(table)
    latex = "\\documentclass{article}\n\\pdfpageheight=11in\n\\pdfpagewidth=8.5in\n\\begin{document}\n" + table + "\\end{document}"
    with open(filename + ".tex", 'w') as f:
        f.write(latex)
    cmd = ['pdflatex', '-interaction', 'nonstopmode', filename + ".tex"]
    proc = subprocess.Popen(cmd)
    proc.communicate()

    retcode = proc.returncode
    if not retcode == 0:
        os.unlink(filename + ".pdf")
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

    os.unlink(filename + ".tex")
    os.unlink(filename + ".log")
    os.unlink(filename + ".aux")

