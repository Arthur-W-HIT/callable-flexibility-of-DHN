import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'portable_dispatch'))
import common as c
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import energy_figure_style as style
from PIL import Image,ImageChops
style.set_energy_style(base_size=8.2,axis_label_size=9.2,legend_size=8.)
COLORS=style.color_order(7)
OUT=Path(__file__).resolve().parent/'regenerated';OUT.mkdir(exist_ok=True)
def save(fig,name,sources,caption):style.save_figure(fig,OUT/name,formats=['pdf','png','svg'],dpi=1000)
def baseline():
    p=c.WORK/'outputs/baseline_points.csv';T=pd.read_csv(p).sort_values('ambient_C')
    fig,ax=plt.subplots(1,2,figsize=(7.08,2.5),layout='constrained')
    ax[0].plot(T.ambient_C,T.source_Q_MW*5,'o-',color=COLORS[0],markersize=2.7)
    ax[0].set(xlabel=r'Ambient temperature, $T_{\mathrm{amb}}$ ($^\circ$C)',ylabel=r'Baseline heat power, $Q^0$ (MW$_{\mathrm{th}}$)')
    ax[1].plot(T.ambient_C,T.verified_room_mean_C,'o-',color=COLORS[1],markersize=2.7,label='Mean over five zones')
    ax[1].axhline(20,color='#777777',ls='--',lw=.9);ax[1].set(xlabel=r'Ambient temperature ($^\circ$C)',ylabel=r'Steady indoor temperature ($^\circ$C)')
    ax[1].set_ylim(19.98,20.02);ax[1].legend(loc='upper right')
    save(fig,'E0_converged_baseline',[p],'Corrected-physics 51 baseline points, N=5 heat aggregation; last12h actual mean and convergence checked. Same-model verification, not field validation.')
baseline()
reference=c.WORK/'figures/E0_converged_baseline.png';generated=OUT/'E0_converged_baseline.png'
a=Image.open(reference).convert('RGB');b=Image.open(generated).convert('RGB')
assert a.size==b.size and ImageChops.difference(a,b).getbbox() is None
c.save(c.WORK/'portable_figure_reproduction.json',dict(passed=True,pixel_identical=True,figure='E0_converged_baseline',
    reference_sha256=c.sha(reference),generated_sha256=c.sha(generated),canvas_pixels=list(a.size),
    scope='isolated same-environment figure regeneration from exported baseline data and copied style helper'))
print('Isolated baseline-figure regeneration is pixel-identical')
