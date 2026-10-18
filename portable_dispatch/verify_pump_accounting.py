"""Verify exported pump electricity and contemporaneous billing without Dymola."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
checks=[]
for name,ledger in [('main','fair_reprice_inherited_week.csv'),('matched','fair_reprice_inherited_matched_hold15.csv'),('season','fair_reprice_season_actual.csv')]:
    maps=json.loads((ROOT/'outputs'/({'main':'main_replay_map.json','matched':'matched_replay_map.json','season':'season_replay_map.json'}[name])).read_text(encoding='utf-8'))
    records=pd.read_csv(ROOT/'outputs'/ledger)
    for _,row in records.iterrows():
        data=pd.read_csv(ROOT/'replays'/maps[row.Case]/'actual_15min.csv')
        energy=float(data.ActualPump_MWh.sum())
        cost=float(np.dot(data.ActualPump_MWh,data.Price_CNY_per_MWh))
        heat=float(data.ActualSourceQ_MWth.sum()*.25)
        assert abs(energy-row.Pump_MWh)<1e-8
        assert abs(cost-row.ActualPumpCost_CNY)<1e-6
        assert abs(heat-row.SourceHeat_MWh)<1e-7
        checks.append(dict(ledger=ledger,case=row.Case,pump_MWh=energy,pump_cost_CNY=cost,passed=True))
    baseline=records.loc[records.Case.eq('NoFlex'),'Net_CNY'].iloc[0]
    assert np.max(abs((records.Net_CNY-baseline)-records.IncrementalNet_CNY))<1e-7
target=ROOT/'pump_accounting_verification.json'
target.write_text(json.dumps(dict(passed=True,checks=checks,scope='Exported simulation ledgers; same-interval price billing and no duplicate proxy charge. No physical model rerun.'),indent=2),encoding='utf-8')
print('Pump-energy, contemporaneous-price cost, source heat, and incremental ledgers verified:',len(checks),'cases')
