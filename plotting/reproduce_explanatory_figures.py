"""Redraw only explanatory figures from unchanged, verified model/output files."""
from pathlib import Path
import hashlib,importlib.util,json,sys,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];R1=ROOT
HELPER=ROOT/'plotting/original_programs/energy_figure_style.py'
sys.path.insert(0,str(HELPER.parent))
from energy_figure_style import set_energy_style,DEFAULT_PALETTE as C,cm_to_in
DATA=R1/'figure_refresh_20261014/data';SEASON=R1/'experiments_20261013/dispatch/Season_Service15'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def export(fig,stem):
    files=[]
    for extension in ['pdf','svg','png']:
        target=ROOT/'regenerated/explanatory'/f'{stem}.{extension}';target.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(target,dpi=600,facecolor='white')
        files.append(str(target.resolve()))
    plt.close(fig)
    return files

def axis_style(ax):
    ax.grid(True,color=C['grid'],linewidth=.6,zorder=0)
    ax.tick_params(length=3,width=.7,labelsize=8.5)
    for spine in ax.spines.values():spine.set_linewidth(.7);spine.set_color(C['neutral_dark'])

def topology():
    rows=pd.read_csv(ROOT/'figure_data/explanatory/pipe_node_mapping.csv').to_dict('records')
    xy={'热源':(.65,0),'J1':(2.2,0),'J2':(4.15,0),'J3':(6.1,0),'J4':(8.05,0),
        'B1':(2.2,1.85),'B2':(6.1,1.85),'B3':(10.35,1.55),'B4':(10.35,-1.55),'B5':(4.15,-1.85)}
    set_energy_style(base_size=9,axis_label_size=9,legend_size=8.5,use_grid=False)
    fig,axes=plt.subplots(2,1,figsize=cm_to_in(18.2,13.2))
    fig.subplots_adjust(left=.03,right=.985,bottom=.055,top=.94,hspace=.35)
    for ax,prefix,color,heading in zip(axes,['PS','PR'],[C['contrast'],C['main']],['(a) Supply pipes','(b) Return pipes']):
        ax.set_xlim(-.05,11.45);ax.set_ylim(-2.3,2.45);ax.axis('off')
        ax.text(.02,1.02,heading,transform=ax.transAxes,ha='left',fontsize=9.5)
        for row in rows:
            if not row['pipe'].startswith(prefix):continue
            # Consistent spatial routing for supply and return; arrows follow actual ports.
            a=xy[row['from_node']];b=xy[row['to_node']]
            if abs(a[0]-b[0])>.1 and abs(a[1]-b[1])>.1:
                turn=(8.05,b[1] if prefix=='PS' else a[1]);route=[a,turn,b]
            else:route=[a,b]
            ax.plot(*zip(*route),color=color,lw=1.35,zorder=1)
            lengths=[np.linalg.norm(np.array(v)-u) for u,v in zip(route,route[1:])]
            j=int(np.argmax(lengths));u,v=np.array(route[j]),np.array(route[j+1]);mid=(u+v)/2;delta=(v-u)/np.linalg.norm(v-u)
            ax.annotate('',xy=mid+.16*delta,xytext=mid-.16*delta,
                arrowprops=dict(arrowstyle='-|>',color=color,lw=1.05,mutation_scale=9),zorder=2)
            if abs(u[1]-v[1])<.01:
                label_xy=mid+np.array([0,.28]);ha='center'
            else:label_xy=mid+np.array([.17,0]);ha='left'
            ax.text(*label_xy,row['pipe'],ha=ha,va='center',fontsize=9,color=C['neutral_dark'],
                bbox=dict(facecolor='white',edgecolor='none',pad=.8),zorder=3)
        for name,(x,y) in xy.items():
            if name.startswith('J'):
                ax.scatter([x],[y],s=15,color=C['neutral_dark'],zorder=4)
                ax.text(x-(.27 if name in ['J2','J4'] else 0),y-.34,name,fontsize=9,ha='center',va='center')
            else:
                label='Source' if name=='热源' else name
                ax.add_patch(Rectangle((x-.40,y-.24),.80,.48,facecolor='white',edgecolor=C['neutral_dark'],lw=.8,zorder=5))
                ax.text(x,y,label,ha='center',va='center',fontsize=9,zorder=6)
    return export(fig,'appendix_primary_network_mapping')

def week_drivers():
    d=pd.read_csv(ROOT/'figure_data/explanatory/week_inputs.csv')
    s=pd.read_csv(ROOT/'figure_data/explanatory/week_schedule.csv')
    tr=pd.read_csv(ROOT/'figure_data/explanatory/week_tracking.csv')
    rm=pd.read_csv(ROOT/'figure_data/explanatory/week_room.csv')
    start=pd.Timestamp('2025-01-08');end=start+pd.Timedelta(days=7)
    h0=float(d.Time_h.iloc[0]);h1=h0+168;x=(d.Time_h.to_numpy()-h0)/24;rx=(rm.time_h-h0)/24
    set_energy_style(base_size=8.5,axis_label_size=9,legend_size=8)
    fig,axes=plt.subplots(5,1,sharex=True,figsize=cm_to_in(18.2,20.4))
    fig.subplots_adjust(left=.12,right=.98,top=.96,bottom=.07,hspace=.48)
    titles=['(a) Ambient input','(b) Electricity price and service availability',
        '(c) Source heat-power command','(d) Supply-temperature response','(e) Room-temperature response']
    for ax,title in zip(axes,titles):
        axis_style(ax);ax.set_title(title,loc='left',fontsize=9,pad=5)
    axes[0].plot(x,d.ToutRaw_C,color=C['neutral_mid'],lw=1,ls=(0,(4,2)),label='Raw weather')
    axes[0].plot(x,d.ToutDispatch_C,color=C['main'],lw=1.2,label='Used in dispatch')
    axes[0].set_ylabel('Ambient temperature\n(°C)');axes[0].legend(loc='lower right',ncol=2)
    flag=d.PeakSupportFlag.to_numpy().astype(bool);edges=np.r_[0,np.where(flag[1:]!=flag[:-1])[0]+1,len(flag)]
    for a,b in zip(edges[:-1],edges[1:]):
        if flag[a]:axes[1].axvspan(a/96,b/96,color=C['positive'],alpha=.30,lw=0,zorder=1)
    axes[1].step(x,d.Price_CNY_per_MWh,where='post',color=C['main'],lw=1.1,zorder=3)
    axes[1].set_ylabel('Electricity price\n(CNY/MWh)');axes[1].set_ylim(-15,690)
    axes[1].legend(handles=[Line2D([0],[0],color=C['main'],lw=1.1,label='Electricity price'),
        Patch(facecolor=C['positive'],alpha=.30,label='Service available')],loc='upper left',ncol=2)
    axes[2].step(x,s.DeltaQ_MWth,where='post',color=C['main'],lw=1.15)
    axes[2].axhline(0,color=C['neutral_dark'],lw=.7);axes[2].set_ylabel('Heat-power shift\n(MW$_{th}$)')
    for case,color,style in [('NoFlex',C['neutral_mid'],(0,(4,2))),('Step15min',C['main'],'-')]:
        t=tr.loc[tr.Case.eq(case)].sort_values('time_h')
        assert len(t)==672
        axes[3].plot((t.time_h-h0)/24,t.source_T_C,color=color,ls=style,lw=1.15,label='NoFlex' if case=='NoFlex' else '15 min')
        axes[4].fill_between(rx,rm[f'{case}_room_min_C'],rm[f'{case}_room_max_C'],color=color,alpha=.12,lw=0)
        axes[4].plot(rx,rm[f'{case}_room_mean_C'],color=color,ls=style,lw=1.15,label='NoFlex' if case=='NoFlex' else '15 min')
    axes[3].set_ylabel('Supply temperature\n(°C)');axes[3].legend(loc='upper right',ncol=2)
    axes[4].axhline(16,color=C['neutral_dark'],ls=':',lw=.9,label='16 °C service floor')
    axes[4].set_ylim(15.7,23.6);axes[4].set_ylabel('Room temperature\n(°C)')
    axes[4].legend(loc='upper right',ncol=3)
    axes[-1].set_xlabel('Time from 8 January 2025 (d)');axes[-1].set_xticks(range(8));axes[-1].set_xlim(0,7)
    files=export(fig,'supplement_week_drivers_and_response')
    return files,dict(date_start=str(start.date()),date_end_inclusive='2025-01-14',selection='Calendar week already used as main week; extracted from the existing continuous season without reoptimization',
        season_time_h=[h0,h1],input_samples=672,tracking_samples_per_case=672,room_samples=len(rm),
        smoothing=False,statistics_changed=False,service_shading='PeakSupportFlag == 1; availability signal, not evidence of causal attribution')

def main():
    topology();week_drivers()
    print('Reproduced Appendix Fig. A.1 and Supplementary Fig. S1')
if __name__=='__main__':main()
