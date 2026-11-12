"""Join local NR90 member-protein counts to Pfam UMAP points and export hover view."""
import csv
import json
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / 'data/Pfam_family_representatives/family_representatives.tsv'


def load_counts(rows):
    with SOURCE.open() as f:
        entries = list(csv.DictReader(f, delimiter='\t'))
    table = {r['pfam_acc']: r for r in entries}
    if len(table) != len(entries):
        raise ValueError('Duplicate Pfam accessions in count source')
    selected = [table[r['id']] for r in rows if r['group'] == 'pfam_reps']
    counts = np.array([int(r['n_proteins_nr90']) for r in selected])
    if np.any(counts <= 0):
        raise ValueError('Every plotted Pfam representative must have a positive membership count')
    return table, counts


def export(coords, labels, rows, out, table, counts, cmap, norm):
    import hashlib
    m = labels == 'pfam_reps'
    selected = [r for r in rows if r['group'] == 'pfam_reps']
    with (out / 'pfam_family_counts.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['pfam_acc', 'pfam_id', 'description', 'n_proteins_nr90', 'n_regions_nr90', 'umap_x', 'umap_y'])
        for row, xy in zip(selected, coords[m]):
            entry = table[row['id']]
            writer.writerow([row['id'], entry['pfam_id'], entry['desc'], entry['n_proteins_nr90'],
                             entry['n_regions_nr90'], float(xy[0]), float(xy[1])])
    palette = {'retain':'#9ba5ad', 'MGYF0000018690':'#009E73', 'PF00106':'#0072B2', 'PF00107':'#CC79A7'}
    groups = []
    for group in dict.fromkeys(labels):
        if group == 'pfam_reps':
            continue
        color = palette.get(group.split('_F_')[0], '#9ba5ad')
        groups.append(dict(name=str(group), color=color, xy=np.round(coords[labels == group], 4).tolist()))
    from matplotlib.colors import to_hex
    reps = []
    for row, xy, count in zip(selected, coords[m], counts):
        entry = table[row['id']]
        reps.append([round(float(xy[0]),4),round(float(xy[1]),4),row['id'],entry['pfam_id'],
                     entry['desc'],int(count),int(entry['n_regions_nr90']),to_hex(cmap(norm(count)))])
    data = json.dumps(dict(groups=groups,reps=reps), separators=(',',':')).replace('<','\\u003c')
    html = '''<!doctype html><meta charset="utf-8"><title>Pfam family membership — ESM2 UMAP</title>
<style>body{margin:0;font:15px system-ui;color:#17212b;background:#fafafa}header{padding:14px 22px}h2{margin:0 0 6px}p{margin:5px 0}#wrap{position:relative;margin:0 20px}canvas{display:block;background:white;border:1px solid #ccc;touch-action:none}#tip{display:none;position:absolute;pointer-events:none;white-space:pre-line;background:#fffffff5;border:1px solid #888;border-radius:5px;padding:10px;max-width:380px;font-size:13px}button,input{font:inherit;margin:5px}#legend span{margin-right:18px}#status{font-size:13px;min-height:20px}</style>
<header><h2>Pfam representatives: member proteins per family</h2>
<p>Hover a Pfam point for its exact NR90 protein count. Drag to pan; scroll to zoom. Coordinates are the existing joint ESM2-35M UMAP.</p>
<div id="legend">Pfam count: <span style="display:inline-block;width:170px;height:12px;background:linear-gradient(to right,GRADIENT)"></span> MINCOUNT → MAXCOUNT (log scale) · R: gray · MGnify F: green · PF00106 F: blue · PF00107 F: purple</div>
<label>Find family <input id="search" placeholder="PF00106 or adh_short"></label><button id="find">Find</button><button id="reset">Reset view</button><div id="status"></div></header>
<div id="wrap"><canvas id="plot" aria-label="Interactive UMAP with hover counts for every Pfam representative"></canvas><div id="tip"></div></div>
<script>const data=DATA;
const canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d'),tip=document.querySelector('#tip'),status=document.querySelector('#status');
let width,height,view,initial,drag=null,selected=null,screen=[],queued=false;
let bounds=[Infinity,-Infinity,Infinity,-Infinity];
for(const g of data.groups)for(const p of g.xy){bounds[0]=Math.min(bounds[0],p[0]);bounds[1]=Math.max(bounds[1],p[0]);bounds[2]=Math.min(bounds[2],p[1]);bounds[3]=Math.max(bounds[3],p[1]);}
for(const p of data.reps){bounds[0]=Math.min(bounds[0],p[0]);bounds[1]=Math.max(bounds[1],p[0]);bounds[2]=Math.min(bounds[2],p[1]);bounds[3]=Math.max(bounds[3],p[1]);}
function xy(p){return [(p[0]-view[0])*view[2]+width/2,height/2-(p[1]-view[1])*view[2]];}
function schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;draw();});}}
function draw(){ctx.clearRect(0,0,width,height);for(const g of data.groups){ctx.fillStyle=g.color;ctx.globalAlpha=g.name==='retain' ? .2 : .3;for(const p of g.xy){const [x,y]=xy(p);if(x>=0&&x<width&&y>=0&&y<height)ctx.fillRect(x,y,1.3,1.3);}}
ctx.globalAlpha=.95;ctx.strokeStyle='#56432a';ctx.lineWidth=.55;screen=[];for(let i=0;i<data.reps.length;i++){const p=data.reps[i],[x,y]=xy(p);screen.push([x,y]);if(x>=0&&x<width&&y>=0&&y<height){ctx.fillStyle=p[7];ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);ctx.fill();ctx.stroke();}}
ctx.globalAlpha=1;ctx.fillStyle='#333';ctx.fillText('UMAP 1 →',width-90,height-12);ctx.fillText('UMAP 2 ↑',12,20);
if(selected!==null){const [x,y]=xy(data.reps[selected]);ctx.strokeStyle='#111';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.stroke();}}
function description(p){return p[2]+' · '+p[3]+'\\n'+p[4]+'\\nMember proteins (NR90): '+p[5].toLocaleString()+'\\nDomain regions (NR90): '+p[6].toLocaleString();}
function resize(){width=Math.max(500,innerWidth-42);height=Math.max(420,innerHeight-220);const d=devicePixelRatio||1;canvas.width=width*d;canvas.height=height*d;canvas.style.width=width+'px';canvas.style.height=height+'px';ctx.setTransform(d,0,0,d,0,0);initial=[(bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2,Math.min((width-70)/(bounds[1]-bounds[0]),(height-70)/(bounds[3]-bounds[2]))];view=[...initial];schedule();}
canvas.onpointerdown=e=>{drag=[e.offsetX,e.offsetY,...view];canvas.setPointerCapture(e.pointerId);tip.style.display='none';};
canvas.onpointerup=()=>drag=null;
canvas.onpointermove=e=>{if(drag){view[0]=drag[2]-(e.offsetX-drag[0])/view[2];view[1]=drag[3]+(e.offsetY-drag[1])/view[2];schedule();return;}let best=-1,dist=64;for(let i=0;i<screen.length;i++){const p=screen[i],d=(p[0]-e.offsetX)**2+(p[1]-e.offsetY)**2;if(d<dist){dist=d;best=i;}}tip.style.display=best<0?'none':'block';if(best>=0){tip.textContent=description(data.reps[best]);tip.style.left=Math.min(e.offsetX+15,width-400)+'px';tip.style.top=Math.max(5,e.offsetY-90)+'px';}};
canvas.onpointerleave=()=>tip.style.display='none';
canvas.addEventListener('wheel',e=>{e.preventDefault();let x=(e.offsetX-width/2)/view[2]+view[0],y=(height/2-e.offsetY)/view[2]+view[1];view[2]=Math.max(initial[2]*.5,Math.min(initial[2]*100,view[2]*Math.exp(-e.deltaY*.001)));view[0]=x-(e.offsetX-width/2)/view[2];view[1]=y-(height/2-e.offsetY)/view[2];tip.style.display='none';schedule();},{passive:false});
document.querySelector('#reset').onclick=()=>{view=[...initial];selected=null;status.textContent='';schedule();};
function find(){const q=document.querySelector('#search').value.trim().toLowerCase();selected=data.reps.findIndex(p=>p[2].toLowerCase()===q||p[3].toLowerCase()===q);if(selected<0){selected=null;status.textContent='Family not found among plotted representatives.';return;}const p=data.reps[selected];view=[p[0],p[1],initial[2]*4];status.textContent=description(p).replaceAll('\\n',' — ');schedule();}
document.querySelector('#find').onclick=find;document.querySelector('#search').onkeydown=e=>{if(e.key==='Enter')find();};window.onresize=resize;resize();
</script>'''
    gradient=','.join(to_hex(cmap(t)) for t in np.linspace(0,1,8))
    html=html.replace('GRADIENT',gradient).replace('MINCOUNT',f'{counts.min():,}').replace('MAXCOUNT',f'{counts.max():,}').replace('DATA',data)
    (out/'landscape_interactive.html').write_text(html)
    provenance=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    count_column='n_proteins_nr90',definition='Member proteins in the supplied NR90 metadata; not domain-region counts or counts of plotted sequences',
                    n_families=len(counts),minimum=int(counts.min()),maximum=int(counts.max()),scale='logarithmic color',
                    coordinates_sha256=hashlib.sha256((out.parent/'umap/coords.npy').read_bytes()).hexdigest())
    (out/'family_counts_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
