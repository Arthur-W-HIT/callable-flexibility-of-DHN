from pathlib import Path

import argparse, ast, json, time

import numpy as np

import pandas as pd

from scipy.optimize import milp, Bounds, LinearConstraint

from scipy.sparse import coo_matrix

import common as c

import dispatch as dp

def options(S,lib):
    rs=[];actions=dp.BASE_ACTION
    for day in range(len(S)//96):
        stage,good,nxt=dp.day_options(S,lib,96*day,actions,dp.CFG,1.)
        tt,ir,ia=np.indices(good.shape)
        data=dict(Step=(tt+96*day+1).ravel(),PrevState=(ir+1).ravel(),
            NextState=np.broadcast_to(nxt+1,good.shape).ravel(),ActionRatio=np.broadcast_to(actions[None,None,:],good.shape).ravel(),Feasible=good.ravel().astype(int))
        for key,value in stage.items():data[key]=value.ravel()
        # The inherited MILP adds a single pump proxy explicitly; exclude it here to avoid double counting.
        data['NetProfit_CNY']=data['NetProfit_CNY']+data['PumpProxyCost_CNY']
        rs.append(pd.DataFrame(data))
    return pd.concat(rs,ignore_index=True)

def solve(case,hold=1,neutral=False,days=7):
    out=c.WORK/'dispatch'/case;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').is_file():return c.load(out/'summary.json')
    lib=dp.Library(c.WORK/'outputs/boundary_H6_floor16.csv')
    S=dp.normalized_inputs(c.WORK/'inputs/main_week.csv',days)
    OPT=options(S,lib);rho=.1**(.25/6)
    C=.5*(S.BoundaryUp15_MW.to_numpy()+S.BoundaryDown15_MW.to_numpy())
    REF=pd.DataFrame(dict(Cnext_MW=np.r_[C[1:],C[-1]],NoFlexState_z_MW=dp.noflex_reference(S,rho)))
    source=Path(__file__).with_name('milp_constraints.py');text=source.read_text(encoding='utf-8-sig');tree=ast.parse(text)
    functions='\n\n'.join(ast.get_source_segment(text,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ['solve_day','check_trajectory'])
    # Retain the tested linear constraint construction and integer-incumbent audit.
    functions=functions.replace("'time_limit':45","'time_limit':120")
    env=dict(np=np,pd=pd,milp=milp,Bounds=Bounds,LinearConstraint=LinearConstraint,coo_matrix=coo_matrix,time=time,
        OPT=OPT,INPUT=S,REF=REF,RHO=rho,STATES=list(range(1,50)))
    exec(compile(functions,str(source)+'[read-only reused constraint functions]','exec'),env)
    pars=dict(hold_steps=hold,neutral=neutral,days=days,rho=rho,terminal_tolerance=.02,rolling_window_steps=24,
        gamma_under=.1,gamma_over=.1,command_heat_neutral_daily_relative_tolerance=.0001,
        solver='scipy.optimize.milp/HiGHS, daily incumbent; no all-week global optimality claim',
        time_limit_per_day_s=120,mip_relative_gap_target=.005,source=str(source),source_sha256=c.sha(source),
        library_sha256=c.sha(lib.path),economic_definition='wear/quadratic plus exactly one pump proxy; actual pump reprice replaces proxy')
    c.save(out/'parameters.json',pars);S.to_csv(out/'inputs.csv',index=False,encoding='utf-8-sig')
    rs=[];info=[];z=0.;ir=1;tic=time.monotonic()
    for day in range(days):
        marker=out/f'day{day+1}.json'
        if marker.is_file():
            saved=c.load(marker);rr=saved['trajectory'];meta=saved['solver'];z=rr[-1]['State_z_MW'];ir=int(rr[-1]['NextState'])
        else:
            rr,z,ir,meta=env['solve_day'](day,z,ir,rs,neutral,hold,True,.1)
            c.save(marker,dict(trajectory=rr,solver=meta))
        rs.extend(rr);info.append(meta)
        print(case,'day',day+1,'solver',meta['solver_status'],'gap',meta['mip_gap'],flush=True)
    df,summary=env['check_trajectory'](rs,neutral,True,.1)
    base=OPT[(OPT.ActionRatio==0)&(OPT.PrevState==1)].sort_values('Step').NetProfit_CNY.sum()
    summary['incremental_profit_with_proxy_CNY']=summary['total_profit_singlePumpProxy_CNY']-float(base)
    summary.update(case=case,elapsed_s=time.monotonic()-tic,solver_days=info,actual_heat_neutrality='requires enthalpy replay; command constraint alone is not physical balance')
    df.to_csv(out/'schedule.csv',index=False,encoding='utf-8-sig');c.save(out/'summary.json',summary)
    return summary

def reprice(cases,pump_factor=1.,fuel_factor=1.,price_factor=1.,suffix='base'):
    tables={key:pd.read_csv(c.WORK/'replays'/value/'actual_15min.csv') for key,value in cases.items()}
    nf=tables['NoFlex'];p0=dp.pmax(nf.ActualSourceQ_MWth.to_numpy()).sum(axis=-1)
    cfg=dict(dp.CFG);cfg['coalPrice_CNY_per_GJ']*=fuel_factor;rs=[];output=[]
    for name,T in tables.items():
        assert len(T)==len(nf)
        replay_metrics=c.load(c.WORK/'replays'/cases[name]/'metrics.json')
        main_steps=int(round((replay_metrics['main_stop_s']-replay_metrics['eval_start_s'])/900))
        price=T.Price_CNY_per_MWh.to_numpy();center=float(price[:main_steps].mean())
        # Paid recovery must not change the optimization input's price mean.
        price=center+price_factor*(price-center)
        data=dp.economics(T.ActualSourceQ_MWth.to_numpy(),price,T.ActionRatio.to_numpy(),T.PeakSupportFlag.to_numpy(),cfg)
        agce=cfg['AGC_energy_CNY_per_MWh']*np.maximum(data['P_MW']-p0,0)*.25*T.PeakSupportFlag.to_numpy()
        pump=T.ActualPump_MWh.to_numpy()*cfg['auxElec_price_CNY_per_MWh']*pump_factor
        net=data['EnergyRevenue_CNY']-data['FuelCost_CNY']-data['FlexCost_CNY']+data['AGCAvailabilityPay_CNY']+agce-pump
        feasible=np.isfinite(net)&(T.ActualSourceQ_MWth.to_numpy()>=dp.PLANT['Qh_min']-1e-6)&(T.ActualSourceQ_MWth.to_numpy()<=dp.PLANT['Qh_max']+1e-6)
        # Preserve violations; never price a NaN as zero or silently discard physical plant violations.
        assert feasible.all(),(name,'actual CHP bounds violated',int((~feasible).sum()))
        row=dict(Case=name,SourceHeat_MWh=float(T.ActualSourceQ_MWth.sum()*.25),Pump_MWh=float(T.ActualPump_MWh.sum()),
            Revenue_CNY=float(data['EnergyRevenue_CNY'].sum()),Fuel_CNY=float(data['FuelCost_CNY'].sum()),
            FlexWearQuadratic_CNY=float(np.sum(data['FlexCost_CNY'])),AGC_CNY=float((data['AGCAvailabilityPay_CNY']+agce).sum()),
            ActualPumpCost_CNY=float(pump.sum()),Net_CNY=float(net.sum()),PlantViolationSteps=0,
            pump_factor=pump_factor,fuel_factor=fuel_factor,price_factor=price_factor,accounting='measured source heat and pumps; no proxy double count')
        row.update(MainNet_CNY=float(net[:main_steps].sum()),RecoveryNet_CNY=float(net[main_steps:].sum()),
            RecoverySourceHeat_MWh=float(T.ActualSourceQ_MWth.iloc[main_steps:].sum()*.25),
            RecoveryPumpCost_CNY=float(pump[main_steps:].sum()),RecoveryHours=(len(T)-main_steps)*.25,
            PriceCenter_CNY_per_MWh=center,price_center_definition='original command-horizon mean, unchanged by charged recovery')
        rs.append(row)
        trace=T.copy()
        for key,val in data.items():
            if key!='PumpProxyCost_CNY':trace[key]=val
        trace['Net_CNY']=net;trace['Case']=name;trace['ActualPumpCost_CNY']=pump
        output.append(trace)
    base=next(r for r in rs if r['Case']=='NoFlex')
    for r in rs:
        r['IncrementalNet_CNY']=r['Net_CNY']-base['Net_CNY'];r['SourceHeatDifference_MWh']=r['SourceHeat_MWh']-base['SourceHeat_MWh']
        r['SourceHeatDifferenceFraction']=r['SourceHeatDifference_MWh']/base['SourceHeat_MWh']
    p=c.WORK/'outputs'/('fair_reprice_'+suffix+'.csv');c.csvout(p,rs)
    if suffix=='base':pd.concat(output).to_csv(c.WORK/'outputs/fair_actual_ledger_15min.csv',index=False,encoding='utf-8-sig')
    return rs
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='Reproduced_Neutral15')
    parser.add_argument('--hold',type=int,default=1)
    parser.add_argument('--days',type=int,default=7)
    parser.add_argument('--neutral',action='store_true')
    opts=parser.parse_args()
    solve(opts.case,hold=opts.hold,neutral=opts.neutral,days=opts.days)
