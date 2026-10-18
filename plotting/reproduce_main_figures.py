"""Reuse the original plotting bodies with the frozen revised CSVs."""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plotting/original_programs'))
DATA=ROOT/'figure_data/main';OUT=ROOT/'regenerated/main';OUT.mkdir(parents=True,exist_ok=True)
specs=json.loads((ROOT/'plotting/figure_specs.json').read_text())
for i,rec in enumerate(specs):
    p=ROOT/'plotting/original_programs'/rec['program']
    spec=importlib.util.spec_from_file_location('original_plot_'+str(i),p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.PACKAGE_DIR=OUT;m.RUN_DIR=OUT;m.FIG_DIR=OUT/'figures';m.DATA_DIR=OUT/'data'
    m.FIG_DIR.mkdir(exist_ok=True);m.DATA_DIR.mkdir(exist_ok=True)
    mapping={'SOURCE_CSV':DATA/'baseline.csv' if 'fig3_' in p.name else DATA/'action_block_cases.csv' if 'fig4_' in p.name else DATA/'boundary.csv',
        'OVERALL_CSV':DATA/'action_block_overall.csv','BOUNDARY_CSV':DATA/'boundary.csv','WEEKLY_DIR':DATA/'weekly',
        'WEEKLY_SUMMARY_CSV':DATA/'weekly/weekly_decision_step_dispatch_summary.csv',
        'WEEKLY_OCCUPANCY_CSV':DATA/'weekly/occupancy_state_trajectory_by_decision_step.csv','SCHEDULE_STEP15':DATA/'weekly/schedule_step15min.csv',
        'PUMP_CASE_CSV':DATA/'pump_case.csv','PUMP_SENSITIVITY_CSV':DATA/'pump_sensitivity_fixed_path.csv',
        'TRACKING_CSV':DATA/'tracking.csv','ROOM_SERIES_CSV':DATA/'room_series.csv','ROOM_SUMMARY_CSV':DATA/'room_summary.csv'}
    for key,value in mapping.items():
        if hasattr(m,key):setattr(m,key,value)
    original=m.save_figure
    def export(fig,stem,*a,_save=original,**kw):
        if Path(stem).name=='corrected_power_boundary_80MWx5':
            ax=fig.axes[1];top=max(float(np.nanmax(np.asarray(l.get_ydata(),float))) for l in ax.lines if len(l.get_ydata()))
            if top>ax.get_ylim()[1]:ax.set_ylim(ax.get_ylim()[0],np.ceil(top/20)*20+20)
        if Path(stem).name=='supply_temperature_cap_filter':
            # Preserve the original layout; separate two inherited annotation collisions.
            for text in fig.axes[1].texts:
                if text.get_text()=='(b)':text.set_x(-.03)
            for text in fig.axes[2].texts:
                if text.get_text() in ['0.89','1.00']:
                    x,y=text.get_position();text.set_position((x,y-.025))
            # Leave space between the upper numeric label and the top frame.
            fig.axes[2].set_ylim(0,1.10)
        fig.canvas.draw()
        kw['formats']=('png','pdf','svg')
        return _save(fig,stem,*a,**kw)
    m.save_figure=export
    if hasattr(m,'set_energy_style'):m.set_energy_style(base_size=8.,axis_label_size=8.7,legend_size=7.4)
    getattr(m,rec['function'])()
    print('Reproduced Fig.',rec['figure'],flush=True)
