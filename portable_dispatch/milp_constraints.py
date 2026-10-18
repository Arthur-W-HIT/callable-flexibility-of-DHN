def solve_day(day,zinit,runinit,past,neutral,hold,terminal=True,gamma=.10):
 start=day*96;end=start+96
 e=OPT[(OPT.Step>start)&(OPT.Step<=end)&(OPT.Feasible==1)].copy().reset_index(drop=True)
 e['j']=np.arange(len(e));n=len(e);N=n+96
 by=[e[e.Step==t+1] for t in range(start,end)]
 cols=[];rows=[];vals=[];lower=[];upper=[]
 def row(idx,v,lo,hi):
  k=len(lower);rows.extend([k]*len(idx));cols.extend(idx);vals.extend(v);lower.append(lo);upper.append(hi)
 # Flow conservation in every run-state node; initial state carried between days.
 for t in range(96):
  for s in STATES:
   outgoing=by[t][by[t].PrevState==s].j.to_numpy()
   incoming=by[t-1][by[t-1].NextState==s].j.to_numpy() if t else np.array([],int)
   target=float(t==0 and s==runinit)
   row(list(outgoing)+list(incoming),[1.]*len(outgoing)+[-1.]*len(incoming),target,target)
  ids=by[t].j.to_numpy();inc=by[t].StateDelta_MW.to_numpy()
  zz=[n+t];coeff=[1.]
  if t:zz.append(n+t-1);coeff.append(-RHO)
  target=RHO*zinit if t==0 else 0.
  row(zz+list(ids),coeff+list(-inc),target,target)
  # Every rolling window includes executed prior-day calls.
  global_t=start+t;first=max(0,global_t-23)
  for sgn in [-1,1]:
   fixed=sum(max(sgn*r['DeltaQ_MWth'],0) for r in past if first<=r['Step']-1<start)
   inds=[];vs=[]
   for j in range(max(0,first-start),t+1):
    inds.extend(by[j].j.tolist());vs.extend(np.maximum(sgn*by[j].DeltaQ_MWth.to_numpy(),0).tolist())
   budget=gamma*INPUT.Qref_MWth.iloc[first:global_t+1].sum()-fixed
   row(inds,vs,-np.inf,budget)
  # Hold aRatio over the declared decision interval, retaining 15min state updates.
  if hold>1 and t%hold:
   for a in sorted(OPT.ActionRatio.unique()):
    ids=by[t][by[t].ActionRatio==a].j.tolist();pr=by[t-1][by[t-1].ActionRatio==a].j.tolist()
    row(ids+pr,[1.]*len(ids)+[-1.]*len(pr),0,0)
 if terminal:
  c=REF.Cnext_MW.iloc[end-1];z0=REF.NoFlexState_z_MW.iloc[end-1]
  row([N-1],[1.],z0-.02*c,z0+.02*c)
 if neutral:
  tolerance=.0001*INPUT.Qref_MWth.iloc[start:end].sum()*.25
  row(e.j.tolist(),(e.DeltaQ_MWth*.25).tolist(),-tolerance,tolerance)
 lo=np.zeros(N);hi=np.ones(N)
 lo[n:]=-.9*REF.Cnext_MW.iloc[start:end].to_numpy();hi[n:]=.9*REF.Cnext_MW.iloc[start:end].to_numpy()
 cost=np.zeros(N)
 pump=2*.35*.25*INPUT.Price_CNY_per_MWh.iloc[e.Step.astype(int).to_numpy()-1].to_numpy()*e.ActionRatio.abs().to_numpy()
 baseprofit=np.array([by[t][by[t].ActionRatio==0].NetProfit_CNY.iloc[0] for t in range(96)])
 cost[:n]=-e.NetProfit_CNY.to_numpy()+baseprofit[e.Step.astype(int).to_numpy()-start-1]+pump
 A=coo_matrix((vals,(rows,cols)),shape=(len(lower),N)).tocsc()
 tic=time.monotonic()
 res=milp(cost,integrality=np.r_[np.ones(n),np.zeros(96)],bounds=Bounds(lo,hi),constraints=LinearConstraint(A,np.array(lower),np.array(upper)),options={'time_limit':45,'mip_rel_gap':.005,'disp':False})
 elapsed=time.monotonic()-tic
 info={'day':day+1,'solver_status':int(res.status),'solver_message':res.message,'elapsed_s':elapsed,'variables':N,'binary_variables':n,'rows':len(lower),'mip_gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None,'objective_gain_CNY':float(-res.fun) if res.fun is not None else None}
 if res.x is None:raise RuntimeError(('No MILP incumbent',info))
 xx=res.x
 ax=A@xx
 violation=max(float(np.maximum(np.array(lower)-ax,0).max()),float(np.maximum(ax-np.array(upper),0).max()),float(np.maximum(lo-xx,0).max()),float(np.maximum(xx-hi,0).max()))
 assert violation<1e-4 and np.max(np.abs(xx[:n]-np.rint(xx[:n])))<1e-5,(info,violation)
 info['max_linear_violation']=violation
 selected=e[xx[:n]>.5].sort_values('Step')
 assert len(selected)==96 and selected.Step.nunique()==96
 rs=[];z=zinit;ir=runinit
 for _,r in selected.iterrows():
  r=r.to_dict();assert r['PrevState']==ir
  z=RHO*z+r['StateDelta_MW'];ir=int(r['NextState']);t=int(r['Step'])-1
  r['State_z_MW']=z;r['State_x']=z/REF.Cnext_MW.iloc[t]
  r['PumpProxyCost_CNY']=2*.35*.25*float(INPUT.Price_CNY_per_MWh.iloc[t])*abs(r['ActionRatio'])
  r['NetProfit_singlePumpProxy_CNY']=r['NetProfit_CNY']-r['PumpProxyCost_CNY']
  rs.append(r)
 assert abs(z-xx[-1])<1e-5
 return rs,z,ir,info

def check_trajectory(rs,neutral,terminal,gamma):
 df=pd.DataFrame(rs);n=len(df);dq=df.DeltaQ_MWth.to_numpy()
 q=INPUT.Qref_MWth.iloc[:n].to_numpy();z=df.State_z_MW.to_numpy()
 under=np.array([np.maximum(-dq[max(0,t-23):t+1],0).sum()/q[max(0,t-23):t+1].sum() for t in range(n)])
 over=np.array([np.maximum(dq[max(0,t-23):t+1],0).sum()/q[max(0,t-23):t+1].sum() for t in range(n)])
 ends=np.arange(95,n,96);tx=(z[ends]-REF.NoFlexState_z_MW.iloc[ends].to_numpy())/REF.Cnext_MW.iloc[ends].to_numpy()
 heat_by_day=[float(dq[i:i+96].sum()*.25) for i in range(0,n,96)]
 assert max(abs(df.State_x))<=.90001 and max(under)<=gamma+1e-6 and max(over)<=gamma+1e-6
 if terminal:assert max(abs(tx))<=.020001
 if neutral:
  for k,h in enumerate(heat_by_day):assert abs(h)<=.0001*q[k*96:(k+1)*96].sum()*.25+1e-5
 result={'total_profit_singlePumpProxy_CNY':float(df.NetProfit_singlePumpProxy_CNY.sum()),'net_heat_difference_MWh':float(dq.sum()*.25),'daily_heat_difference_MWh':heat_by_day,'maximum_state_abs':float(max(abs(df.State_x))),'maximum_terminal_deviation_relative_NoFlex':float(max(abs(tx))),'maximum_rolling_under_ratio':float(max(under)),'maximum_rolling_over_ratio':float(max(over)),'active_steps':int(sum(df.ActionRatio!=0))}
 return df,result
