"""
created matt_dumont 
on: 10/12/24
"""

from komanawa.simple_farm_model.speed_test import run_model
from memory_profiler import profile

@profile
def run_mem_test(nyr, instep, opt_mode):
    run_model(nyr, instep, opt_mode)

if __name__ == '__main__':
    for nsteps in [10, 50]:
        run_mem_test(1,  nsteps, 'coarse')
        run_mem_test(10,  nsteps, 'coarse')
