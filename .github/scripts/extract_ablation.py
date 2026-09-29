"""Extract Figure 5 geometry from a pdftocairo SVG of manuscript page 6.

Usage: python extract_ablation.py page6.svg
The resulting CSV contains digitized approximations, not original run data.
"""
import argparse
import xml.etree.ElementTree as ET
import re,csv
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('svg',type=Path)
args=parser.parse_args()
root=ET.parse(args.svg).getroot()
colors=['rgb(76.','rgb(24.','rgb(30.','rgb(58.']; methods=['ReF-HIL','w/o value shaping','w/o action fence','w/o both']; tasks=['Push-T','Cap unscrewing','Gear assembly','Plug insertion','Dual-branch cable routing']
def points(el):
 nums=list(map(float,re.findall(r'-?\d*\.?\d+(?:e[-+]?\d+)?',el.get('d','')))); pts=list(zip(nums[::2],nums[1::2])); tr=el.get('transform')
 if tr:
  a,b,c,d,e,f=map(float,re.findall(r'-?\d*\.?\d+(?:e[-+]?\d+)?',tr)); pts=[(a*x+c*y+e,b*x+d*y+f) for x,y in pts]
 return pts
bars=[]; errors=[]
for el in root.iter():
 if not el.tag.endswith('path'): continue
 fill=el.get('fill',''); pts=points(el)
 if not pts: continue
 xs,ys=zip(*pts)
 if any(fill.startswith(c) for c in colors) and 5<max(xs)-min(xs)<7:
  row=0 if max(ys)<100 else 1; bars.append((row,(min(xs)+max(xs))/2,max(ys)-min(ys),next(i for i,c in enumerate(colors) if fill.startswith(c))))
 if el.get('stroke','').startswith('rgb(19.999') and len(pts)==2 and abs(xs[0]-xs[1])<.001:
  errors.append((xs[0],sum(ys)/2,abs(ys[0]-ys[1])/2))
rows=[]
for row in [0,1]:
 for ti,task in enumerate(tasks):
  for mi,method in enumerate(methods):
   x=51.779589+39.60198*ti+6.93*mi
   bar=[b for b in bars if b[0]==row and b[3]==mi and abs(b[1]-x)<.03]
   if not bar: assert row==0 and mi==3 and ti in [1,4]
   err=[e for e in errors if abs(e[0]-x)<.03 and (e[1]<95 if row==0 else e[1]>95)]
   assert len(err)==1,(row,task,method,err)
   scale=46.184109 if row==0 else 46.184 # graph units per 100 percentage points
   value=(bar[0][2] if bar else 0)/scale*100
   rows.append(dict(task=task,method=method,metric='autonomous_success_percent' if row==0 else 'normalized_episode_length_percent',mean=round(value,1),sample_sd=round(err[0][2]/scale*100,1),provenance='Digitized from manuscript Figure 5 vector geometry; approximate'))
out=Path(__file__).resolve().parents[2]/'assets/ablation.csv'
with out.open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
for row in rows: print(row['task'],row['method'],row['metric'],row['mean'],row['sample_sd'])
