"""Redraw manuscript Figure 5 from the digitized, reader-downloadable CSV."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
rows=list(csv.DictReader((ROOT/'assets/ablation.csv').open()))
methods=['ReF-HIL','w/o value shaping','w/o action fence','w/o both']
tasks=['Push-T','Cap unscrewing','Gear assembly','Plug insertion','Dual-branch cable routing']
colors=['#1c6250','#548ca8','#c18a4b','#a5adb1']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#182e2b','axes.labelcolor':'#526460','xtick.color':'#526460','ytick.color':'#526460','svg.fonttype':'none'})
fig,axes=plt.subplots(2,1,figsize=(12,7.2),sharex=True)
for ax,metric,title in zip(axes,['autonomous_success_percent','normalized_episode_length_percent'],['Autonomous success rate (%) ↑','Normalized episode length (%) ↓']):
 for mi,(method,color) in enumerate(zip(methods,colors)):
  data=[next(r for r in rows if r['task']==t and r['method']==method and r['metric']==metric) for t in tasks]
  values=[float(r['mean']) for r in data]; sd=[float(r['sample_sd']) for r in data]; xs=np.arange(5)+(mi-1.5)*.185
  ax.bar(xs,values,.165,label=method,color=color,zorder=3,yerr=sd,error_kw={'elinewidth':1,'capsize':2.4,'ecolor':'#344b44'})
  for x,y in zip(xs,values):
   if y==0: ax.text(x,2,'0',ha='center',fontsize=9,color='#69766b')
 ax.set_title(title,loc='left',fontsize=12,fontweight='semibold',pad=13); ax.set_ylim(0,112); ax.set_yticks([0,25,50,75,100]); ax.grid(axis='y',color='#e6ece8',linewidth=.8,zorder=0); ax.tick_params(axis='both',length=0,pad=8)
 for spine in ax.spines.values(): spine.set_visible(False)
axes[-1].set_xticks(range(5),['Push-T','Cap\nunscrewing','Gear\nassembly','Plug\ninsertion','Dual-branch\ncable routing'])
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=4,frameon=False,fontsize=11,handlelength=1.3,columnspacing=2)
fig.subplots_adjust(top=.87,bottom=.1,left=.065,right=.99,hspace=.36)
fig.savefig(ROOT/'assets/figures/ablation.svg',bbox_inches='tight',metadata={'Date':None})
fig.savefig(ROOT/'assets/figures/ablation.png',bbox_inches='tight',dpi=180)
