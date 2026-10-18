"""Season-basis daily grid DP with exact carried state and hard terminal/rolling checks."""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
import common as c

CFG=c.load(c.WORK/'outputs/economic_configuration.json')['cfg']
PLANT=c.load(c.WORK/'outputs/economic_configuration.json')['plant']
BASE_ACTION=np.array(CFG['actionRatioGrid'],float).reshape(-1)

class Library:
    def __init__(self,path):
        self.path=Path(path);data=pd.read_csv(path)
        assert len(data)==2346 and (data.boundary_status=='usable').all(),('Full, diagnosed unified library required',len(data))
        self.temps=np.sort(data.T_amb_C.unique());self.durs=np.r_[np.sort(data.tau_act_min.unique()),360.]
        self.f={}
        for sign,direction in [(1,'upward'),(-1,'downward')]:
            for key,col in [('B','accepted_delta_Q_MW_80MWx5'),('DT','accepted_delta_T_supply_abs_C')]:
                arr=data[data.direction==direction].pivot(index='T_amb_C',columns='tau_act_min',values=col).reindex(index=self.temps,columns=self.durs[:-1]).to_numpy()
                if key=='B':arr=np.abs(arr)
                assert np.isfinite(arr).all() and (arr>=0).all()
                self.f[(key,sign)]=RegularGridInterpolator((self.temps,self.durs),np.column_stack([arr,np.zeros(len(arr))]),bounds_error=True)
        br=data[(data.direction=='upward')&(data.tau_act_min==15)].sort_values('T_amb_C')
        self.baseQ=br.baseline_heat_MW_80MW.to_numpy()*5
        self.baseTs=br.baseline_supply_C.to_numpy()
        verified=pd.read_csv(c.WORK/'outputs/baseline_points.csv').sort_values('ambient_C')
        self.flow=verified.flow_scale.to_numpy()
    def lookup(self,key,T,d,sign):
        t,du=np.broadcast_arrays(np.asarray(T),np.asarray(d))
        return self.f[(key,sign)](np.column_stack([np.clip(t.ravel(),-35,15),np.clip(du.ravel(),15,360)])).reshape(t.shape)

def pmax(q):
    q=np.asarray(q,float);powers=[]
    for u,share in zip(PLANT['units'],PLANT['heatShare']):
        x=np.array(u['Qh_upper']);y=np.array(u['P_upper'])
        order=np.argsort(x);powers.append(PchipInterpolator(x[order],y[order],extrapolate=False)(q*share))
    return np.stack(powers,axis=-1)

def economics(q,price,ratio,peak,cfg=CFG,pump_multiplier=1.):
    pu=pmax(q);power=pu.sum(axis=-1)
    hr=PchipInterpolator(np.array(cfg['P_hr']).reshape(-1),np.array(cfg['HR_hr']).reshape(-1))
    cp=np.clip(pu,min(np.array(cfg['P_hr']).reshape(-1)),max(np.array(cfg['P_hr']).reshape(-1)))
    h=hr(cp);h[...,1]*=cfg['unit2HeatRateScale']
    fuel=.25*(pu*h).sum(axis=-1)/1000*cfg['coalPrice_CNY_per_GJ']
    revenue=power*np.asarray(price)*.25
    flex=.25*(cfg['wearCost_full']*np.abs(ratio)*(1+cfg['unit2FlexCostScale'])+cfg['flexQuadraticCost_full']*np.asarray(ratio)**2)
    pump=.25*cfg['nUnit']*cfg['pumpExtraMW_full']*cfg['auxElec_price_CNY_per_MWh']*np.abs(ratio)*pump_multiplier
    avail=cfg['nUnit']*cfg['AGC_availability_CNY_per_h']*.25*np.asarray(peak)
    return dict(P_MW=power,P1_MW=pu[...,0],P2_MW=pu[...,1],FuelCost_CNY=fuel,EnergyRevenue_CNY=revenue,FlexCost_CNY=flex,PumpProxyCost_CNY=pump,AGCAvailabilityPay_CNY=avail)


def normalized_inputs(path,days=None):
    S=pd.read_csv(path)
    if days is not None:S=S.iloc[:96*days].copy()
    S=S.reset_index(drop=True)
    n=len(S);assert n%96==0
    S['Step']=np.arange(1,n+1);S['Day']=np.arange(n)//96+1;S['StepInDay']=np.arange(n)%96+1
    return S

def day_options(S,lib,start,actions,cfg,pump_multiplier):
    n=96;nr=49;na=len(actions);ts=np.arange(start,start+n)
    signs=np.r_[0,-np.ones(24),np.ones(24)].astype(int)
    runs=np.r_[0,np.arange(1,25),np.arange(1,25)]
    sg=np.sign(actions).astype(int)
    run=np.where(sg[None,:]==0,0,np.where(signs[:,None]==sg[None,:],runs[:,None]+1,1))
    ns=np.where(sg[None,:]==0,0,np.where(sg[None,:]<0,run,run+24))
    ns=np.broadcast_to(ns,(nr,na));validrun=run<=24
    shape=(n,nr,na);T=S.ToutDispatch_C.to_numpy()[ts,None,None]
    T=np.broadcast_to(T,shape);dur=np.broadcast_to(np.maximum(run*15,15),shape)
    Bup=lib.lookup('B',T,dur,1);Bdn=lib.lookup('B',T,dur,-1)
    bd=np.where(sg[None,None,:]<0,Bdn,Bup)
    b15=np.where(sg[None,None,:]<0,S.BoundaryDown15_MW.to_numpy()[ts,None,None],S.BoundaryUp15_MW.to_numpy()[ts,None,None])
    dq=actions[None,None,:]*b15
    phi=dq/np.maximum(bd,np.finfo(float).eps)
    dt=phi*np.where(sg[None,None,:]<0,lib.lookup('DT',T,dur,-1),lib.lookup('DT',T,dur,1))
    eff=dt+S.TsRefDelta_C.to_numpy()[ts,None,None]
    tot=np.where(eff<0,lib.lookup('DT',T,dur,-1),lib.lookup('DT',T,dur,1))
    qref=S.Qref_MWth.to_numpy();q=qref[ts,None,None]+dq
    band=np.where(T<=-14,.05,np.where(T<=-10,.07,.10))
    stage=economics(q,S.Price_CNY_per_MWh.to_numpy()[ts,None,None],actions[None,None,:],S.PeakSupportFlag.to_numpy()[ts,None,None],cfg,pump_multiplier)
    p0=pmax(qref[ts]).sum(axis=-1)[:,None,None]
    agce=cfg['AGC_energy_CNY_per_MWh']*np.maximum(stage['P_MW']-p0,0)*.25*S.PeakSupportFlag.to_numpy()[ts,None,None]
    stage['AGCEnergyPay_CNY']=agce
    stage['NetProfit_CNY']=stage['EnergyRevenue_CNY']-stage['FuelCost_CNY']-stage['FlexCost_CNY']-stage['PumpProxyCost_CNY']+stage['AGCAvailabilityPay_CNY']+agce
    feasible=validrun[None,:,:]&(abs(dq)<=bd+1e-7)&(abs(dq)<=band*qref[ts,None,None]+1e-7)&(abs(eff)<=tot+1e-7)&np.isfinite(stage['NetProfit_CNY'])
    feasible&=(q>=PLANT['Qh_min']-1e-7)&(q<=PLANT['Qh_max']+1e-7)
    C=.5*(S.BoundaryUp15_MW.to_numpy()+S.BoundaryDown15_MW.to_numpy())
    refshift=qref[ts]-qref[np.minimum(ts+1,len(S)-1)]
    stage.update(DeltaQ_MWth=dq,DeltaTsrc_C=dt,StateDelta_MW=C[ts,None,None]*phi+refshift[:,None,None],
        Qh_MWth=q,EffectiveBoundary_MW=bd,EffectiveDuration_min=np.broadcast_to(run*15,shape),RefShift_MW=np.broadcast_to(refshift[:,None,None],shape))
    stage={name:np.broadcast_to(value,shape) for name,value in stage.items()}
    return stage,feasible,ns

def noflex_reference(S,rho):
    q=S.Qref_MWth.to_numpy();z=0.;out=[]
    for t in range(len(q)):
        z=rho*z+q[t]-q[min(t+1,len(q)-1)];out.append(z)
    return np.array(out)

def blocks(stage,feasible,nextstate,hold,rho):
    nd=96//hold;nr,na=nextstate.shape
    offset=np.zeros((nd,nr,na,hold));profit=np.zeros((nd,nr,na));okay=np.ones((nd,nr,na),bool)
    endstate=np.empty((nd,nr,na),int)
    for d in range(nd):
        st=np.broadcast_to(np.arange(nr)[:,None],(nr,na)).copy();a=np.broadcast_to(np.arange(na)[None,:],(nr,na))
        z=np.zeros((nr,na))
        for k in range(hold):
            t=d*hold+k;safe=np.clip(st,0,nr-1)
            okay[d]&=(st<nr)&feasible[t,safe,a]
            z=rho*z+stage['StateDelta_MW'][t,safe,a];offset[d,:,:,k]=z
            profit[d]+=stage['NetProfit_CNY'][t,safe,a];st=nextstate[safe,a]
        endstate[d]=np.clip(st,0,nr-1)
    return offset,profit,okay,endstate

def exact_future_intervals(offset,okay,endstate,powers,limits,target,eps):
    """Backward union of feasible real-state intervals; independent of value-grid rounding."""
    nd,nr,na=okay.shape;future=[[[] for _ in range(nr)] for _ in range(nd+1)]
    for ir in range(nr):future[nd][ir]=[(target-eps-1e-8,target+eps+1e-8)]
    decay=float(powers[-1]);max_intervals=1
    for d in range(nd-1,-1,-1):
        for ir in range(nr):
            pieces=[]
            for a in np.flatnonzero(okay[d,ir]):
                lo=float(np.max((-limits[d]-1e-8-offset[d,ir,a])/powers))
                hi=float(np.min((limits[d]+1e-8-offset[d,ir,a])/powers))
                if lo>hi:continue
                shift=offset[d,ir,a,-1]
                for left,right in future[d+1][endstate[d,ir,a]]:
                    l=max(lo,(left-shift)/decay);h=min(hi,(right-shift)/decay)
                    if l<=h:pieces.append((l,h))
            merged=[]
            for left,right in sorted(pieces):
                if merged and left<=merged[-1][1]+1e-10:merged[-1]=(merged[-1][0],max(right,merged[-1][1]))
                else:merged.append((left,right))
            future[d][ir]=merged;max_intervals=max(max_intervals,len(merged))
    return future,max_intervals

def in_intervals(z,intervals):return any(left-1e-8<=z<=right+1e-8 for left,right in intervals)

def solve_day_dp(S,start,zinit,runinit,lib,hold,grid,actions,cfg,pump_multiplier,rho,nfref):
    stage,feasible,ns=day_options(S,lib,start,actions,cfg,pump_multiplier)
    offset,profit,okay,endstate=blocks(stage,feasible,ns,hold,rho)
    nr,na=ns.shape;nd=96//hold
    C=.5*(S.BoundaryUp15_MW.to_numpy()+S.BoundaryDown15_MW.to_numpy());n=len(S)
    Cnext=C[np.minimum(np.arange(start,start+96)+1,n-1)]
    target=nfref[start+95];eps=cfg['terminalTolerance']*Cnext[-1]
    zmax=cfg['stateSlack']*C.max();zg=np.linspace(-zmax,zmax,grid);dz=zg[1]-zg[0]
    def closest(z):
        scaled=np.nan_to_num(np.clip((z+zmax)/dz,0,grid-1),nan=0.,posinf=grid-1,neginf=0.)
        return np.floor(scaled+.5).astype(int)
    V=np.full((nd+1,grid,nr),-np.inf)
    allowed=abs(zg-target)<=eps+1e-8
    V[nd,allowed,:]=-cfg['terminalPenalty_CNY']*abs((zg[allowed]-target)/Cnext[-1])[:,None]
    ztracepow=rho**np.arange(1,hold+1);indices=np.arange(nr)[None,:]
    all_limits=np.array([cfg['stateSafetyMargin']*np.minimum(C[start+d*hold:start+(d+1)*hold],Cnext[d*hold:(d+1)*hold]) for d in range(nd)])
    future,max_intervals=exact_future_intervals(offset,okay,endstate,ztracepow,all_limits,target,eps)
    if not in_intervals(zinit,future[0][runinit]):raise RuntimeError(('No exact feasible state continuation',start,zinit,runinit))
    for d in range(nd-1,-1,-1):
        for a in range(na):
            traces=zg[:,None,None]*ztracepow[None,None,:]+offset[d,:,a,:][None,:,:]
            limits=cfg['stateSafetyMargin']*np.minimum(C[start+d*hold:start+(d+1)*hold],Cnext[d*hold:(d+1)*hold])
            good=okay[d,:,a][None,:]&np.all(abs(traces)<=limits[None,None,:]+1e-8,axis=-1)
            z2=traces[:,:,-1];iz=closest(z2)
            if d==nd-1:good&=(abs(z2-target)<=eps+1e-8)
            value=profit[d,:,a][None,:]+V[d+1,iz,endstate[d,:,a][None,:]]
            V[d]=np.maximum(V[d],np.where(good,value,-np.inf))
    z=zinit;ir=runinit;records=[]
    for d in range(nd):
        traces=z*ztracepow[None,:]+offset[d,ir,:,:]
        limits=cfg['stateSafetyMargin']*np.minimum(C[start+d*hold:start+(d+1)*hold],Cnext[d*hold:(d+1)*hold])
        good=okay[d,ir,:]&np.all(abs(traces)<=limits[None,:]+1e-8,axis=-1)
        if d==nd-1:good&=(abs(traces[:,-1]-target)<=eps+1e-8)
        good&=np.array([in_intervals(traces[a,-1],future[d+1][endstate[d,ir,a]]) for a in range(na)])
        value=profit[d,ir,:]+V[d+1,closest(traces[:,-1]),endstate[d,ir,:]]
        # A true feasible interval can contain no finite nearest value-grid cell.
        # Preserve exact feasibility and use the closest finite value only as an
        # objective estimate, never as a substitute for the real-state checks.
        if not np.isfinite(value[good]).any():
            zero=int(np.flatnonzero(actions==0)[0]);baseline_future=float(profit[d+1:,0,zero].sum())
            for a in np.flatnonzero(good):
                column=V[d+1,:,endstate[d,ir,a]];finite=np.flatnonzero(np.isfinite(column))
                value[a]=profit[d,ir,a]+(column[finite[np.argmin(abs(zg[finite]-traces[a,-1]))]] if len(finite) else baseline_future)
        value=np.where(good,value,-np.inf)
        if not np.isfinite(value).any():raise RuntimeError(('DP exact reconstruction has no viable continuation',start,d,grid,z))
        a=int(np.argmax(value));ratio=float(actions[a])
        for k in range(hold):
            t=d*hold+k;row={name:float(np.broadcast_to(vals,(96,nr,na))[t,ir,a]) for name,vals in stage.items()}
            assert feasible[t,ir,a]
            z=rho*z+row['StateDelta_MW'];ir=int(ns[ir,a])
            row.update(Step=start+t+1,ActionRatio=ratio,State_z_MW=z,State_x=z/Cnext[t],NextState=ir)
            records.append(row)
    assert abs(z-target)<=eps+1e-7
    return records,z,ir,dict(grid_points=grid,action_grid=actions.tolist(),daily_decisions=nd,
        run_states=nr,terminal_relative_deviation=(z-target)/Cnext[-1],
        exact_feasibility_intervals=True,max_interval_count=max_intervals,
        value_at_grid_initial=float(V[0,closest(zinit),runinit]) if np.isfinite(V[0,closest(zinit),runinit]) else None)

def check(S,df,rho,cfg,hold):
    q=S.Qref_MWth.to_numpy();dq=df.DeltaQ_MWth.to_numpy();z=0.;err=0.
    for t,r in df.iterrows():
        z=rho*z+r.StateDelta_MW;err=max(err,abs(z-r.State_z_MW))
    assert err<1e-6 and abs(df.State_x).max()<=cfg['stateSafetyMargin']+1e-6
    refs=noflex_reference(S,rho);C=.5*(S.BoundaryUp15_MW.to_numpy()+S.BoundaryDown15_MW.to_numpy())
    ends=np.arange(95,len(df),96);cn=C[np.minimum(ends+1,len(C)-1)]
    term=(df.State_z_MW.to_numpy()[ends]-refs[ends])/cn
    assert abs(term).max()<=cfg['terminalTolerance']+1e-6
    up=pd.Series(np.maximum(dq,0)).rolling(24,min_periods=1).sum().to_numpy()/pd.Series(q).rolling(24,min_periods=1).sum().to_numpy()
    down=pd.Series(np.maximum(-dq,0)).rolling(24,min_periods=1).sum().to_numpy()/pd.Series(q).rolling(24,min_periods=1).sum().to_numpy()
    assert up.max()<=cfg['gammaOver']+1e-6 and down.max()<=cfg['gammaUnder']+1e-6
    grouped=df.ActionRatio.to_numpy().reshape(-1,hold)
    assert (np.ptp(grouped,axis=1)<1e-10).all()
    return dict(incremental_profit_with_proxy_CNY=float(df.NetProfit_CNY.sum()),state_recurrence_error_MW=err,
        max_abs_state=float(abs(df.State_x).max()),max_terminal_relative_deviation=float(abs(term).max()),
        max_rolling_under_ratio=float(down.max()),max_rolling_over_ratio=float(up.max()),
        command_net_heat_difference_MWh=float(dq.sum()*.25),active_steps=int((df.ActionRatio!=0).sum()),
        same_direction_max_duration_min=float(df.EffectiveDuration_min.max()))

def solve(case,inputfile,hold=1,grid=81,action_refine=False,residual=.1,fuel_factor=1.,price_factor=1.,pump_factor=1.,days=None,library_path=None):
    out=c.WORK/'dispatch'/case;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').is_file():return c.load(out/'summary.json')
    lib=Library(library_path or c.WORK/'outputs/boundary_H6_floor16.csv');S=normalized_inputs(inputfile,days)
    cfg=copy.deepcopy(CFG);cfg['coalPrice_CNY_per_GJ']*=fuel_factor
    price=S.Price_CNY_per_MWh.to_numpy();S['Price_CNY_per_MWh']=price.mean()+price_factor*(price-price.mean())
    S.to_csv(out/'inputs.csv',index=False,encoding='utf-8-sig')
    actions=BASE_ACTION if not action_refine else np.unique(np.r_[BASE_ACTION,(BASE_ACTION[:-1]+BASE_ACTION[1:])/2])
    rho=residual**(.25/6);nf=noflex_reference(S,rho);z=0.;ir=0;rs=[];info=[];tic=time.monotonic()
    c.save(out/'parameters.json',dict(hold_steps=hold,requested_grid=grid,action_grid=actions.tolist(),rho=rho,residual_6h=residual,
        fuel_factor=fuel_factor,price_factor=price_factor,pump_factor=pump_factor,library_sha256=c.sha(lib.path),
        terminal_tolerance=cfg['terminalTolerance'],gamma_under=cfg['gammaUnder'],gamma_over=cfg['gammaOver'],
        weather_input=str(inputfile),weather_input_sha256=c.sha(inputfile),solver_source_sha256=c.sha(__file__),solver='daily grid DP; exact feasible-interval guard and continuous state reconstruction; explicit hard terminal and rolling verification'))
    for day in range(len(S)//96):
        marker=out/f'day{day+1}.json'
        if marker.is_file():
            saved=c.load(marker);records=saved['trajectory'];meta=saved['solver'];z=records[-1]['State_z_MW'];ir=int(records[-1]['NextState'])
        else:
            start=96*day;error=None
            for factor in [1,2,4,8]:
                ngrid=(grid-1)*factor+1
                try:records,zn,rn,meta=solve_day_dp(S,start,z,ir,lib,hold,ngrid,actions,cfg,pump_factor,rho,nf);error=None;break
                except RuntimeError as exc:error=repr(exc)
            if error:raise RuntimeError((case,day,error))
            z=zn;ir=rn;c.save(marker,dict(trajectory=records,solver=meta))
        rs.extend(records);info.append(meta)
        if day%7==0 or day==len(S)//96-1:print(case,'day',day+1,'/',len(S)//96,'grid',meta['grid_points'],flush=True)
    df=pd.DataFrame(rs);summary=check(S,df,rho,cfg,hold)
    baseline=[]
    for day in range(len(S)//96):
        st,_,_=day_options(S,lib,day*96,BASE_ACTION,cfg,pump_factor);zero=int(np.where(BASE_ACTION==0)[0][0]);baseline.extend(st['NetProfit_CNY'][:,0,zero])
    summary['NoFlex_profit_CNY']=float(np.sum(baseline));summary['incremental_profit_with_proxy_CNY']-=summary['NoFlex_profit_CNY']
    summary.update(case=case,solver_days=info,elapsed_s=time.monotonic()-tic,day_count=len(S)//96,
                   needs_dynamic_replay=True,heat_reference='command proxy; actual source enthalpy and pump electricity reported separately after replay')
    df.to_csv(out/'schedule.csv',index=False,encoding='utf-8-sig');c.save(out/'summary.json',summary)
    return summary




if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='Reproduced_Service15')
    parser.add_argument('--input',default='main_week')
    parser.add_argument('--hold',type=int,default=1)
    parser.add_argument('--grid',type=int,default=641)
    parser.add_argument('--days',type=int,default=7)
    options=parser.parse_args()
    solve(options.case,c.WORK/'inputs'/f'{options.input}.csv',hold=options.hold,grid=options.grid,days=options.days)
