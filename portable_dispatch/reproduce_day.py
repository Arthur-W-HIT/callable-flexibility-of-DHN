import json,sys,platform,shutil,tempfile
from pathlib import Path
import numpy as np,pandas as pd,scipy
import common as c
import dispatch as dp
ROOT=c.WORK
reference=c.load(Path(__file__).parent/'reference.json')
expected=pd.read_csv(Path(__file__).parent/'reference_schedule.csv')
scratch=ROOT/'regenerated';scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='dispatch_check_',dir=scratch) as directory:
    work=Path(directory).resolve();assert work.is_relative_to(scratch.resolve())
    (work/'outputs').mkdir()
    for name in ['economic_configuration.json','baseline_points.csv','boundary_H6_floor16.csv']:
        shutil.copy2(ROOT/'outputs'/name,work/'outputs'/name)
    c.WORK=work
    try:
        r=dp.solve('Fresh_Service15_Day1',ROOT/'inputs/main_week.csv',hold=1,grid=641,days=1)
        actual=pd.read_csv(work/'dispatch/Fresh_Service15_Day1/schedule.csv')
        assert abs(r['incremental_profit_with_proxy_CNY']-reference['expected_proxy_gain_CNY'])<1e-6
        assert np.array_equal(actual.ActionRatio.to_numpy(),expected.ActionRatio.to_numpy())
    finally:c.WORK=ROOT
c.save(ROOT/'portable_dispatch_reproduction.json',dict(passed=True,fresh_optimization=True,cached_schedule_reused=False,proxy_gain_CNY=r['incremental_profit_with_proxy_CNY'],
    actions_identical=True,terminal_relative_deviation=r['max_terminal_relative_deviation'],state_error_MW=r['state_recurrence_error_MW'],
    scope='standalone scheduling subset only, not independent physical-model validation or complete Dymola reproduction',
    runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,os=platform.platform())))
print('Portable one-day dispatch matches reference, including exact action sequence')
