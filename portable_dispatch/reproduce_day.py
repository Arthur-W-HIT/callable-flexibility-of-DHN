import json,sys,platform
from pathlib import Path
import numpy as np,pandas as pd,scipy
import common as c
import dispatch as dp
reference=c.load(Path(__file__).parent/'reference.json')
r=dp.solve('Portable_Service15_Day1',c.WORK/'inputs/main_week.csv',hold=1,grid=641,days=1)
actual=pd.read_csv(c.WORK/'dispatch/Portable_Service15_Day1/schedule.csv')
expected=pd.read_csv(Path(__file__).parent/'reference_schedule.csv')
assert abs(r['incremental_profit_with_proxy_CNY']-reference['expected_proxy_gain_CNY'])<1e-6
assert np.array_equal(actual.ActionRatio.to_numpy(),expected.ActionRatio.to_numpy())
c.save(c.WORK/'portable_dispatch_reproduction.json',dict(passed=True,proxy_gain_CNY=r['incremental_profit_with_proxy_CNY'],
    actions_identical=True,terminal_relative_deviation=r['max_terminal_relative_deviation'],state_error_MW=r['state_recurrence_error_MW'],
    scope='standalone scheduling subset only, not independent physical-model validation or complete Dymola reproduction',
    runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,os=platform.platform())))
print('Portable one-day dispatch matches reference, including exact action sequence')
