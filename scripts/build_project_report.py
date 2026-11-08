#!/usr/bin/env python3
"""Build the portable IITP year-two report from verified, saved experiment data.

No training/evaluation is performed. Figures and report content are embedded in
one HTML file; the input extraction and figures remain available for audit.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/analysis/report_20261107'
BASELINES = ROOT / 'artifacts/analysis/baselines_20261103/report_comparison_tables.json'
TASKS = ('mintrec2', 'mintrec20', 'mer6')
NAMES = {'mintrec2': 'MIntRec-2', 'mintrec20': 'MIntRec-20', 'mer6': 'MER2023-SEMI'}
STAGES = ('original', 'lm_only', 'final')
STAGE_LABELS = {'original': 'Original task SFT', 'lm_only': 'LM recovery', 'final': 'Final recovery · LM + encoders'}
COLORS = {'original': '#64748b', 'lm_only': '#087f8c', 'final': '#ba6817'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':11,
                     'axes.labelsize':9, 'xtick.labelsize':9, 'ytick.labelsize':9,
                     'axes.edgecolor':'#cbd5e1', 'axes.spines.top':False,
                     'axes.spines.right':False, 'grid.color':'#e5eaf0',
                     'grid.linewidth':.65, 'svg.fonttype':'none',
                     'figure.facecolor':'white', 'axes.facecolor':'white'})


def esc(v):
    return html.escape(str(v), quote=True)


def pct(v):
    return f'{100*v:.2f}%'


def billions(v, precision=3):
    return f'{v/1e9:.{precision}f}B'


def num(v):
    return f'<span class="num">{esc(v)}</span>'


def table(headers, rows, caption=None, cls='', row_classes=None):
    head=''.join(f'<th scope="col">{h}</th>' for h in headers)
    classes = row_classes if row_classes is not None else [''] * len(rows)
    assert len(classes) == len(rows)
    body=''.join(f'<tr class="{esc(row_cls)}">'+''.join(f'<td>{c}</td>' for c in row)+'</tr>'
                 for row, row_cls in zip(rows, classes))
    cap=f'<caption>{caption}</caption>' if caption else ''
    return f'<div class="table-wrap {cls}"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def inline_svg(fig, key, title):
    matplotlib.rcParams['svg.hashsalt']=key
    out=io.StringIO()
    fig.savefig(out,format='svg',bbox_inches='tight',metadata={'Date':None, 'Title':title})
    fig.savefig(ART/f'{key}.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    svg=out.getvalue()[out.getvalue().index('<svg'):]
    (ART/f'{key}.svg').write_text(svg)
    ids=set(re.findall(r'\bid="([^"]+)"',svg))
    for ident in sorted(ids,key=len,reverse=True):
        pref=key+'-'+ident
        svg=svg.replace(f'id="{ident}"',f'id="{pref}"')
        svg=svg.replace(f'#{ident}"',f'#{pref}"').replace(f'#{ident})',f'#{pref})')
    svg=svg.replace('<svg ', f'<svg role="img" aria-label="{esc(title)}" ',1)
    return svg


def validation_figure(primary):
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for task,ax in zip(TASKS,axs):
        for stage in STAGES:
            r=primary[stage]['runs'][task]
            vals=np.array(r['curves']['validation'])
            ax.plot(vals[:,0],vals[:,2],marker='o',markersize=3.2,lw=1.8,
                    color=COLORS[stage],label=STAGE_LABELS[stage])
            b=r['best_validation']
            ax.scatter([b['epoch']],[b['score']],s=74,facecolors='white',
                       edgecolors=COLORS[stage],linewidths=1.7,zorder=6)
        ax.set(title=NAMES[task],xlabel='Epoch within this stage',xlim=(-.08,5.12))
        ax.set_ylabel('Validation weighted F1' if task=='mer6' else 'Validation accuracy')
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis='y'); ax.set_xticks(range(6))
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    return inline_svg(fig,'validation_comparison','Validation performance across three tasks and three training stages')


def smoothed_train(points,window=51):
    a=np.array(points,dtype=float)
    if len(a)<window:return a[:,0],a[:,1]
    kernel=np.ones(window)/window
    return a[window-1:,0],np.convolve(a[:,1],kernel,mode='valid')


def loss_figure(primary,stage):
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for task,ax in zip(TASKS,axs):
        r=primary[stage]['runs'][task]
        tr=np.array(r['curves']['train_sft'])
        va=np.array(r['curves']['validation'])
        take=max(1,len(tr)//600)
        ax.plot(tr[::take,0],np.maximum(tr[::take,1],1e-6),color='#b8cbd2',alpha=.25,lw=.6)
        x,y=smoothed_train(tr)
        ax.plot(x,np.maximum(y,1e-6),color='#087f8c',lw=1.7,label='Train SFT (51-update mean)')
        ax.plot(va[:,0],va[:,1],color='#ba6817',marker='o',markersize=3.6,lw=1.7,label='Full-validation SFT')
        ax.axvline(r['best_validation']['epoch'],color='#64748b',lw=1,ls='--',label='Selected by validation score')
        ax.set(title=NAMES[task],xlabel='Epoch within this stage',ylabel='Supervised loss (log scale)',xlim=(-.08,5.12),yscale='log')
        ax.set_xticks(range(6));ax.grid(axis='y',which='major')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    return inline_svg(fig,'loss_'+stage,'Supervised training and validation loss: '+STAGE_LABELS[stage])


CSS=r'''
:root{--ink:#172535;--muted:#546477;--line:#dce4e9;--teal:#087f8c;--teal-light:#eaf5f5;--amber:#a85c10;--amber-light:#fff7e9;--paper:#fff;--bg:#f4f6f8;--navy:#122b3b}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;color:var(--ink);background:var(--bg);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.65}a{color:var(--teal);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#055c66}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #18a2b3;outline-offset:4px}.page{max-width:1450px;margin:auto;padding:34px 46px 70px}.topline{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:26px}.brand{font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:750;color:var(--muted)}.actions{display:flex;gap:10px}.button{border:1px solid #cad6de;border-radius:8px;background:#fff;color:var(--ink);padding:8px 13px;font-size:12px;font-weight:650}.button:hover{background:#eaf5f5;border-color:#8bbbc2}.layout{display:grid;grid-template-columns:184px minmax(0,1fr);gap:40px}.sidebar{position:sticky;top:25px;align-self:start;font-size:12px;padding-top:14px}.sidebar .label{font-size:10px;letter-spacing:.13em;font-weight:800;text-transform:uppercase;color:#7a8997}.sidebar ol{list-style:none;padding:0;margin:15px 0 25px}.sidebar li{margin:4px 0}.sidebar a{display:block;text-decoration:none;padding:8px 10px;border-left:2px solid transparent;color:var(--muted)}.sidebar a:hover,.sidebar a[aria-current=true]{background:#e6eff1;border-left-color:var(--teal);color:#075d68}.side-note{border-top:1px solid var(--line);padding:14px 10px 0;color:var(--muted);font-size:11px;line-height:1.65}.report{min-width:0}header.hero{padding:37px 40px 32px;background:var(--navy);color:#fff;border-radius:14px}.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:750;color:#a5dce1}.hero h1{font-size:clamp(28px,3.15vw,43px);letter-spacing:-.035em;line-height:1.16;margin:15px 0 18px;max-width:760px}.hero .lead{font-size:16px;color:#d8e6ee;max-width:730px;margin:0}.meta{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:25px;font-size:11px;color:#b9d0dd}.summary-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0 20px}.stat{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:19px 21px}.stat strong{display:block;font-size:28px;letter-spacing:-.04em;line-height:1.2;color:#123749}.stat span{display:block;font-size:12px;color:var(--muted);margin-top:5px}.standfirst{font-size:16px;margin:22px 3px 31px;line-height:1.75}.section{padding:31px 34px 34px;margin:22px 0;background:var(--paper);border:1px solid var(--line);border-radius:12px;scroll-margin-top:22px}.section-kicker{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);font-weight:800;margin-bottom:5px}.section h2{font-size:26px;letter-spacing:-.025em;line-height:1.25;margin:0 0 16px}.section h3{font-size:18px;line-height:1.35;margin:27px 0 12px;letter-spacing:-.015em}.section h4{font-size:15px;margin:18px 0 7px}.section p{margin:10px 0 15px}.section ul,.section ol{padding-left:21px;margin:10px 0 17px}.section li{margin:6px 0}.muted,.caption{color:var(--muted)}.small{font-size:12px}.caption{font-size:12px;line-height:1.65;margin:11px 0 20px}.tag{display:inline-block;padding:3px 8px;border-radius:5px;font-size:10px;line-height:1.35;font-weight:750;letter-spacing:.02em;white-space:nowrap}.tag.good{color:#05626c;background:#e3f3f1}.tag.open{color:#855015;background:#fff0d8}.tag.context{color:#546477;background:#edf1f4}.callout{padding:16px 19px;border-left:3px solid var(--teal);background:var(--teal-light);border-radius:0 7px 7px 0;margin:18px 0;font-size:13px}.callout p:last-child{margin-bottom:0}.callout p:first-child{margin-top:0}.callout.amber{background:var(--amber-light);border-color:#d19a40}.callout.neutral{background:#f1f5f7;border-color:#a9b8c3}.roadmap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:20px 0}.year{border:1px solid var(--line);border-radius:8px;padding:15px 13px;font-size:12px;line-height:1.6;background:#f9fbfc}.year.current{border-color:#66a8b3;background:#edf7f7}.year .date{font-size:11px;color:var(--muted);font-weight:700;letter-spacing:.05em}.year h3{font-size:15px;margin:5px 0 9px}.year p{margin:5px 0 0}.year .contextline{font-size:10px;margin-top:10px;color:var(--muted);border-top:1px solid #dbe5e8;padding-top:8px}.table-wrap{overflow-x:auto;margin:15px 0 17px;max-width:100%;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;font-size:12px;line-height:1.55;white-space:normal}caption{text-align:left;padding:11px 13px;font-size:11px;color:var(--muted);background:#f9fbfc;border-bottom:1px solid var(--line)}th{background:#f2f6f8;text-align:left;font-size:11px;font-weight:750;vertical-align:bottom;padding:11px 12px;color:#425567}td{padding:12px;vertical-align:top;border-top:1px solid #e7edf0}tbody tr:hover{background:#f8fcfc}td .sub{display:block;font-size:10px;color:var(--muted);margin-top:4px;line-height:1.45}.num{font-variant-numeric:tabular-nums;white-space:nowrap}.emphasis{background:#eef7f6}.best{font-weight:800;color:#066574}.method-table td:first-child{min-width:175px}.method-table td:not(:first-child){white-space:nowrap}.results-table td:first-child{min-width:160px}.results-table th:not(:first-child),.results-table td:not(:first-child){text-align:right}.formula{display:block;padding:13px 16px;margin:12px 0;background:#f3f6f8;border:1px solid #e0e8ed;border-radius:7px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;line-height:1.9;overflow-wrap:anywhere}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:22px}.two-col>*{min-width:0}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.method-card{border:1px solid var(--line);border-radius:8px;padding:16px 18px;background:#fbfcfd}.method-card h4{margin:0 0 7px;font-size:14px}.method-card p{font-size:12px;margin:0}.method-card.featured{border-color:#92c1c5;background:#eff8f8}.method-card:last-child{grid-column:1/-1}.workflow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0 12px}.flow-step{padding:19px 17px;border:1px solid var(--line);border-radius:8px;background:#f7fafb}.flow-step.step2{background:#edf7f7;border-color:#9ecbd0}.flow-step.step3{background:#fff7eb;border-color:#e6c797}.flow-step .step-no{font-size:10px;letter-spacing:.08em;font-weight:800;color:var(--muted);text-transform:uppercase}.flow-step h4{font-size:15px;margin:6px 0}.flow-step p{font-size:12px;margin:8px 0}.flow-end{display:inline-block;margin-top:8px;font-size:11px;font-weight:800;color:#075e68}.flow-step.step3 .flow-end{color:#915108}figure{margin:18px 0 22px;padding:15px 12px 9px;border:1px solid var(--line);border-radius:9px;background:white;overflow:hidden}figure svg{display:block;width:100%;height:auto}figcaption{font-size:11px;line-height:1.65;color:var(--muted);padding:6px 8px 7px}.figure-title{font-weight:750;font-size:13px;padding:2px 8px 0;margin-bottom:5px}.stage-controls{display:flex;flex-wrap:wrap;gap:7px;margin:13px 0 0}.stage-controls button{border:1px solid #cddbe0;background:#fff;padding:8px 12px;border-radius:6px;font-size:12px;color:#4d6170}.stage-controls button[aria-pressed=true]{background:#173d4a;border-color:#173d4a;color:#fff}.stage-controls button:hover{border-color:#087f8c}.curve-notes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.curve-note{font-size:12px;border-top:2px solid #a7cbd0;padding-top:11px}.curve-note strong{display:block;margin-bottom:6px}details{margin:16px 0;border:1px solid var(--line);border-radius:7px;background:#fafcfd}summary{cursor:pointer;padding:12px 15px;font-size:12px;font-weight:750;color:#315465}details>div{padding:0 15px 15px;font-size:12px}details .table-wrap{margin-top:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;line-height:1.8;background:#eef3f5;padding:15px;border-radius:6px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}.source-list{padding-left:20px;font-size:11px;color:var(--muted)}.source-list li{margin:11px 0}.source-list strong{color:#304758}.footer{font-size:11px;color:var(--muted);padding:5px 3px 0;line-height:1.8}.no-break{break-inside:avoid}.todo-list td:first-child{width:80px}.todo-list td:nth-child(2){width:35%}.legend-dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}.benchmark-table table{min-width:680px}.benchmark-table th:nth-child(1){width:25%}.benchmark-table th:nth-child(2){width:43%}.benchmark-table th:nth-child(3){width:17%}.benchmark-table th:last-child,.benchmark-table td:last-child{text-align:right;white-space:nowrap}.reference-row{background:#f6f8fa;color:var(--muted)}.literature-sources .sub{display:block;font-size:11px;color:var(--muted);margin-top:3px}.report-end{border-top:1px solid var(--line);padding-top:15px;margin-top:25px;font-size:12px;color:var(--muted)}
@media(min-width:1600px){body{font-size:16px}}
@media(max-width:1100px){.page{padding:22px 25px 50px}.layout{grid-template-columns:145px minmax(0,1fr);gap:24px}.section{padding:26px 25px}.hero{padding:30px!important}.roadmap{grid-template-columns:repeat(2,minmax(0,1fr))}.workflow{grid-template-columns:1fr}.flow-step{padding:15px 18px}.flow-step p{margin:5px 0}.two-col{gap:17px}.curve-notes{grid-template-columns:1fr}}
@media(max-width:760px){.page{padding:17px 14px 35px}.layout{display:block}.sidebar{position:static;padding:0;margin-bottom:15px}.sidebar .label,.side-note{display:none}.sidebar ol{display:flex;gap:3px;flex-wrap:wrap;margin:0}.sidebar a{padding:6px 9px;border:1px solid #dce5e9;border-radius:6px;background:white;font-size:10px}.topline{align-items:flex-start;margin-bottom:18px}.brand{font-size:10px;max-width:175px}.actions{gap:6px}.button{padding:7px 9px;font-size:10px}.hero{padding:26px 23px!important}.hero .lead{font-size:14px}.hero h1{font-size:31px}.section{padding:24px 19px}.section h2{font-size:23px}.summary-strip{gap:8px}.stat{padding:14px 11px}.stat strong{font-size:23px}.stat span{font-size:10px}.standfirst{font-size:14px}.roadmap,.method-grid,.two-col{grid-template-columns:1fr}.method-card:last-child{grid-column:auto}.year{padding:13px 15px}.year p{font-size:12px}.meta{gap:6px 14px}.method-table table{min-width:615px}.results-table table{min-width:540px}th,td{padding:10px}.section p{font-size:14px}.small,.section p.small,.caption{font-size:11px}figure{padding:9px 3px 4px}figure svg{min-width:660px}.chart-scroll{overflow-x:auto}.stage-controls button{font-size:11px;padding:7px 9px}.section li{font-size:14px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@page{size:A4 landscape;margin:13mm 15mm}
@media print{body{background:white;color:#172535;font-size:10px;line-height:1.5}.page{max-width:none;padding:0}.topline{margin-bottom:12px}.actions,.sidebar,.stage-controls{display:none!important}.layout{display:block}.hero{background:#eef3f5!important;color:#172535!important;padding:22px 26px!important;border:1px solid #d0dce3}.hero h1{font-size:29px}.hero .lead,.hero .meta,.hero .eyebrow{color:#355164}.hero .lead{font-size:12px}.meta{margin-top:15px}.summary-strip{margin:12px 0}.stat{padding:12px 16px}.stat strong{font-size:23px}.stat span{font-size:10px}.standfirst{font-size:12px;margin:15px 0}.section{padding:21px 23px;margin:15px 0;border-radius:4px;box-shadow:none}.section h2{font-size:21px}.section h3{font-size:15px}.section p,.section li{font-size:11px}.roadmap{grid-template-columns:repeat(4,minmax(0,1fr))}.workflow{grid-template-columns:repeat(3,minmax(0,1fr))}.curve-notes{grid-template-columns:repeat(3,minmax(0,1fr))}.two-col,.method-grid{grid-template-columns:1fr 1fr}.method-card:last-child{grid-column:1/-1}table{font-size:10px}th{font-size:9px}td,th{padding:8px 9px}tr,figure,.workflow,.roadmap,.callout,.method-card{break-inside:avoid}h2,h3,h4{break-after:avoid}thead{display:table-header-group}.table-wrap{overflow:visible}.method-table table,.results-table table{min-width:0}figure svg{min-width:0;max-height:210px}.chart-scroll{overflow:visible}.loss-figure[hidden]{display:block!important}.loss-figure{break-inside:avoid}details>div{display:block!important}details:not([open])>*:not(summary){display:block!important}summary{list-style:none;padding-bottom:8px}a{color:inherit;text-decoration:none}.source-list{font-size:9px}.footer{font-size:9px}.caption,figcaption{font-size:9px}.tag{font-size:8px}.no-print{display:none!important}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
'''




def literature_comparison_section(data, primary):
    literature = data['literature_comparisons']
    assert set(literature['sources']) == {'M', 'R', 'E'}
    papers = literature['papers']
    panels, ranks = [], {}
    for task in TASKS:
        comparison = literature['tasks'][task]
        external = comparison['external']
        score = 100 * primary['final']['runs'][task]['test_primary']
        rank = 1 + sum(row['score_percent'] > score for row in external)
        ranks[task] = (rank, len(external) + 1)
        ours = {'id':'ours', 'name':'Ours · Activation-pruned LM + encoder adaptation',
                'score_percent':score, 'modalities':'AV' if task == 'mer6' else 'AVT'}
        ordered = sorted(external + [ours], key=lambda row:(-row['score_percent'], row['name'], row['modalities']))
        rows, classes = [], []
        for row in ordered + comparison['reference']:
            own = row['id'] == 'ours'
            reference = row in comparison['reference']
            name = f'<strong>{esc(row["name"])}</strong>' if own else esc(row['name'])
            name += f'<span class="sub">Inputs: {esc(row["modalities"])}</span>'
            if reference:
                name += '<span class="sub">Human reference · excluded from model ranking</span>'
            if own:
                origin = 'This project'
                source = '<a href="#results">Official test · selected by validation</a>'
            else:
                paper = papers[row['original_paper']]
                origin = (f'<a href="{esc(paper["url"])}" target="_blank" rel="noopener">{esc(paper["short_title"])}</a>'
                          f'<span class="sub">{esc(paper["citation"])}</span>')
                source = '<br>'.join(f'<a href="#lit-source-{esc(s["paper"])}">{esc(s["paper"])} · {esc(s["table"])}</a>' for s in row['score_sources'])
            value = num(f'{row["score_percent"]:.2f}%')
            rows.append([name, origin, source, f'<strong>{value}</strong>' if own else value])
            classes.append('emphasis' if own else ('reference-row' if reference else ''))
        metric = 'weighted F1' if task == 'mer6' else 'accuracy'
        ordinal = {1:'1st', 2:'2nd', 3:'3rd'}.get(rank, f'{rank}th')
        panels.append(f'<h3 id="comparison-{task}">{NAMES[task]} · {metric}</h3>'
            + table(['Baseline name / inputs', 'Original method paper', 'Score source', 'Performance'], rows,
                    caption=f'{len(external)} published model configurations + our chosen model. Sorted by performance.',
                    cls='benchmark-table', row_classes=classes)
            + f'<p class="caption">Our model ranks <strong>{ordinal} of {len(external)+1} configurations</strong> in this comparison set. '
              + ('The ≥92% binary-intent target is met.' if task == 'mintrec2' else 'The 20-class comparison is supplementary to the binary-intent target.' if task == 'mintrec20' else 'The top-five benchmark-comparison target is met.') + '</p>')
    summary = ''.join(f'<div class="stat"><strong>{ranks[t][0]} / {ranks[t][1]}</strong><span>{NAMES[t]} · configuration rank</span></div>' for t in TASKS)
    sources = ''.join(f'<li id="lit-source-{key}"><strong>{key}.</strong> <a href="{esc(s["url"])}" target="_blank" rel="noopener">{esc(s["title"])}</a>'
                       f'<span class="sub">{esc(s["citation"])} · {esc(s["location"])}</span></li>' for key,s in literature['sources'].items())
    return '''<section class="section" id="baselines"><div class="section-kicker">06 · Published comparisons</div>
<h2>Top-five placement in the selected benchmark comparisons</h2>
<p>Scores come from three primary sources: the original MIntRec paper, ARL and Emotion-LLaMA. Each row identifies both the original method paper and the paper reporting its benchmark score. A/V/T mean audio/video/text. Our rows use the same chosen architecture and recovery recipe across all three tasks, with separately trained task weights.</p>
<div class="summary-strip">''' + summary + '''</div>
<div class="callout neutral"><strong>Ranking scope.</strong> Ranks count individual published configurations in the named comparison tables, including modality and backbone variants. Identical rows shared by sources are counted once. These are literature comparisons with reported metrics; training compute, seed aggregation and data use are not controlled across papers, and the set is not an exhaustive literature leaderboard.</div>
<ol class="source-list literature-sources">''' + sources + '</ol>' + ''.join(panels) + '''
<details><summary>Baseline inclusion and score provenance</summary><div><ul>
<li>MIntRec Table 4: all ten model configurations and the human reference. ARL Table 1: all six model configurations. Identical MAG-BERT and MulT results are merged, with both sources shown.</li>
<li>ARL Table 9a contributes all four additional 20-class MAG-BERT controls: Curriculum Dropout, OGM, OPM and InfoReg. Table 9b concerns a different dataset and is outside this comparison.</li>
<li>Emotion-LLaMA Table 3 contributes all sixteen configurations, including both AV/AVT versions of the official baseline and Emotion-LLaMA. Higher-scoring variants remain in the ranking.</li>
<li>The MIntRec text-classifier score is the dataset authors’ BERT-based adaptation. TCL-MAP’s binary score is reported by ARL. For MER unimodal rows, the cited original paper describes the encoder; the score comes from the Emotion-LLaMA comparison.</li>
<li>Human references, component ablations and hyperparameter sweeps are outside model ranking. Published baselines are reported results, not local reruns.</li></ul>
<p>The complete numerical catalog and citation metadata are embedded in the downloadable result data.</p></div></details></section>'''


def build(data):
    ART.mkdir(parents=True, exist_ok=True)
    primary = {p['id']:p for p in data['primary']}
    assert set(primary) == set(STAGES)
    for stage in STAGES:
        for task in TASKS:
            r = primary[stage]['runs'][task]
            assert r['test']['logit']['n'] == (834 if task == 'mer6' else 445)
            assert r['training']['seed'] == 42 and len(r['curves']['validation']) == 11
    final = primary['final']['runs']
    assert all(r['test']['logit'][r['primary_metric']] == r['test']['greedy'][r['primary_metric']] for r in final.values())
    params = final['mintrec2']['parameters']
    assert all(r['parameters'] == params for r in final.values())
    original_params = primary['original']['runs']['mintrec2']['parameters']
    reduction = 100 * (1 - params['language']/original_params['language'])
    total_reduction = 100 * (1 - params['total']/original_params['total'])
    validation = validation_figure(primary)
    losses = {s:loss_figure(primary,s) for s in STAGES}
    inv = data['inventory']
    nav = [('overview','01 · Project roadmap'),('targets','02 · Year-two objectives'),('protocol','03 · Tasks & protocol'),('methods','04 · Chosen method'),('results','05 · Experimental results'),('baselines','06 · Published comparisons'),('curves','07 · Training evidence'),('next','08 · Remaining work'),('sources','Sources & reproducibility')]
    navhtml = ''.join(f'<li><a href="#{k}">{label}</a></li>' for k,label in nav)
    body = []
    body.append('''<header class="hero"><div class="eyebrow">IITP Smarthome · KAIST · Efficiency workstream</div>
<h1>Efficient multimodal models<br>for the smart home</h1>
<p class="lead">Activation-based language-model pruning and staged distillation with audio and vision encoder adaptation.</p>
<div class="meta"><span>7 November 2026 · KST</span><span>Official results verified 6 November 2026</span><span>For our lab and technical collaborators</span></div></header>
<div class="summary-strip">''' + ''.join(f'<div class="stat"><strong>{pct(final[t]["test_primary"])}</strong><span>{NAMES[t]} · official test {"weighted F1" if t == "mer6" else "accuracy"}</span></div>' for t in TASKS) + '''</div>
<p class="standfirst">Our chosen model keeps the native audio and vision encoders and reduces the language model to <strong>1.925B parameters</strong> through structured FFN pruning. Two recovery rounds use supervised fine-tuning and teacher distillation, with encoder adapters in the final round. The same architecture and numerical recovery recipe meet the <strong>92% binary-intent threshold</strong> and rank <strong>1st / 1st / 4th</strong> in our selected intent and emotion comparison tables. The model satisfies the confirmed year-two component-size limits.</p>''')
    body.append('''<section class="section" id="overview"><div class="section-kicker">01 · Funding project context</div>
<h2>A four-year roadmap, with efficiency in focus</h2>
<p>The consortium is developing a privacy-preserving, multimodal smart-home system that understands user state and intent and supports personalized, empathetic interaction. Our lab’s focus here is model efficiency; sensing, interaction design and field integration are shared consortium work. <a href="#src-plan">[1–2]</a></p>
<div class="roadmap"><div class="year"><div class="date">2025 · YEAR 1</div><h3>Establish foundations</h3><p>Investigate low-cost inference, weight quantization and multimodal/sensor-state feasibility.</p><div class="contextline">Lab work: LLM quantization and WESAD emotion prediction. Consortium context: 1 linked modality.</div></div>
<div class="year current"><div class="date">2026 · YEAR 2</div><h3>Compress and evaluate</h3><p><strong>LM ≤2B; each modality encoder ≤1B.</strong> Establish task performance and benchmark comparisons; deliver an open-source lightweight model.</p><div class="contextline">Lab presentation: 1 model release; comparison rank ≤5. Consortium context: 2 modalities.</div></div>
<div class="year"><div class="date">2027 · YEAR 3</div><h3>Support integration</h3><p>Develop efficient models as additional modalities, context understanding and personalization are integrated.</p><div class="contextline">Consortium context: 3 modalities; comparison-rank indicator 4. The audited annual table specifies no additional intermediate size cap.</div></div>
<div class="year"><div class="date">2028 · YEAR 4</div><h3>Compress further</h3><p><strong>LM ≤1.5B; each modality encoder ≤0.5B.</strong> Support the integrated solution and final model release.</p><div class="contextline">Consortium context: 5 modalities; comparison-rank indicator 3.</div></div></div>
<p class="caption">Efficiency milestones are grounded in the work-package plan, lab presentation and coordinator correspondence. The modality roadmap supplies project context; it does not imply that this report implements every sensing or integration component.</p>
<div class="callout amber"><strong>KPI/source note: lab deliverables and consortium targets.</strong><p>The formal consortium table lists emotion F1 ≥0.93 in year two and ≥0.95 in year four, with intent accuracy ≥92% in year four. The coordinator’s September guidance assigns our lab the ≥92% MIntRec binary-intent threshold and MER2023-SEMI F1/top-five comparisons for year two. This report establishes those assigned benchmark and size results; the separate consortium emotion KPI remains a reporting-scope clarification. <a href="#src-plan">[1]</a> <a href="#src-email">[3]</a></p></div></section>''')
    target_rows = [
        ['Language model','≤2,000,000,000 parameters',num(f'{params["language"]:,}') + ' <span class="tag good">MET</span>'],
        ['Audio encoder','≤1,000,000,000 parameters',num(f'{params["audio"]:,}') + ' <span class="tag good">MET</span>'],
        ['Vision encoder','≤1,000,000,000 parameters',num(f'{params["vision"]:,}') + ' <span class="tag good">MET</span>'],
        ['Binary intent','Official MIntRec 2-class accuracy ≥92%','<strong>92.13%</strong> <span class="tag good">MET</span>'],
        ['Benchmark comparisons','Top five in the selected intent and emotion comparisons','<strong>Intent 1/15 · MER 4/17</strong> <span class="tag good">MET</span>'],
        ['20-class intent','Supplementary competitive comparison','<strong>78.43% ACC · 1/19</strong> <span class="tag good">EVALUATED</span>'],
        ['Open-source model','One lightweight-model release','<span class="tag open">RELEASE PREPARATION</span>'],
    ]
    body.append('''<section class="section" id="targets"><div class="section-kicker">02 · Year-two objectives</div>
<h2>Independent component limits and task performance</h2>
<p>The coordinator confirmed that the encoder limit applies <strong>per modality</strong>. The selected model therefore prunes the LM and retains the original encoders, each within its own 1B limit. <a href="#src-email">[3]</a></p>''' + table(['Deliverable','Criterion','Verified evidence'],target_rows) + '''
<div class="callout neutral"><strong>Parameter accounting.</strong> Count all learned weights in the classification Thinker: LM embeddings, decoder, normalization and output head, plus audio/vision encoders and their interfaces. Counts describe the dense model after LoRA merging. Unused Talker and Token2Wav speech-generation modules are excluded. The unpruned Qwen2.5-Omni-3B classification Thinker contains <strong>4.703B</strong> parameters; the selected model contains <strong>3.231B</strong> in total. The year-two limits apply to the LM and individual encoders, rather than the total Thinker count.</div></section>''')
    dataset_rows = [[f'<strong>{esc(t["name"])}</strong><span class="sub">{t["classes"]} classes</span>',num(f'{t["train"]:,}'),num(f'{t["validation"]:,}'),num(f'{t["test"]:,}'),'Audio + video' + (' + provided transcript' if t['id'] != 'mer6' else ''),'Weighted F1' if t['id']=='mer6' else 'Accuracy'] for t in data['tasks']]
    body.append('''<section class="section" id="protocol"><div class="section-kicker">03 · Tasks and implementation</div>
<h2>Fixed data splits and full official tests</h2>''' + table(['Dataset / task','Train','Validation','Official test','Inputs','Primary metric'],dataset_rows) + '''
<div class="two-col"><div><h3>MIntRec: intent recognition</h3><p>The <strong>original MIntRec</strong> supplies official train/validation/test splits. Binary and 20-class experiments use the same clips and separate task models. Binary classification directly predicts “express emotions or attitudes” versus “achieve goals” using the official taxonomy; it is not a mapping of our 20-class predictions. This is the original benchmark, distinct from MIntRec2.0.</p></div>
<div><h3>MER2023-SEMI: emotion recognition</h3><p>The labelled 3,373-clip TrainVal set is stratified into <strong>2,698 train / 675 validation</strong> clips with split seed 42. The official SEMI test has <strong>834 clips</strong>. Classes are neutral, angry, happy, sad, worried and surprise. The selected method uses audio/video without an explicit transcript.</p></div></div>
<div class="callout neutral"><strong>MER validation and test populations.</strong> Internal validation and the official SEMI test have different class distributions—for example, “sad” accounts for 77/675 validation clips and 257/834 test clips. Their weighted F1 values are not interchangeable estimates on an identical distribution.</div>
<h3>Media, prompting and evaluation</h3>''' + table(['Item','Implemented protocol'],[
        ['Backbone','BF16 Qwen2.5-Omni-3B Thinker, with native audio and vision encoders; answer-letter classification.'],
        ['Media','Request 2 fps; sample across the clip up to 8 frames, with a 50,176-pixel per-frame budget and preserved aspect ratio. Use full native audio. Duration guards reject clips over 30 seconds for MIntRec or 60 seconds for MER.'],
        ['Prompt','Fixed task instruction plus an enumerated list of answer letters. MIntRec includes the supplied utterance transcript. Request exactly one option letter, without explanation.'],
        ['Supervised target','Correct option letter followed by EOS. Prompt/media tokens are masked from the loss; the gold answer is not part of the input prompt.'],
        ['Logit evaluation','Choose the highest next-token score among allowed class letters. Full-validation accuracy/F1 in this mode selects the checkpoint.'],
        ['Greedy evaluation','Generate without sampling; apply a fixed option-letter parser. Invalid or ambiguous answers remain errors, with no logit fallback. Maximum generation length: 8 tokens.'],
        ['Data separation','Calibration and cached class/feature teacher targets use training clips only. Full validation runs at step 0 and approximately every half epoch. The selected checkpoint is evaluated on the complete official test in both modes.'],
    ]) + '''<details><summary>Example prompt and label conventions</summary><div><pre>Identify the speaker’s communicative intent in this audiovisual clip.
Use the spoken utterance, vocal delivery, and visible context.
Utterance transcript: [provided transcript]
Allowed labels:
A. Express emotions or attitudes [...]
B. Achieve goals [...]
Answer with exactly one option letter and no explanation.</pre><p>Each task has a fixed label order stored with its predictions. The official binary taxonomy assigns <code>Care</code> to emotions/attitudes and <code>Comfort</code> to goals. The 20-class task preserves the original labels, including <code>Apologise</code>. Training seed and split seed are 42 for all primary runs.</p></div></details>
<p class="caption">Author releases: <a href="https://github.com/thuiar/MIntRec" target="_blank" rel="noopener">MIntRec</a> and <a href="https://arxiv.org/html/2406.11161v2" target="_blank" rel="noopener">Emotion-LLaMA’s MER2023-SEMI protocol</a>. Reported results use separate task checkpoints, with task-specific teachers and established AVT/AV inputs.</p></section>''')
    body.append('''<section class="section" id="methods"><div class="section-kicker">04 · Chosen method</div>
<h2>Prune the LM, then adapt the complete understanding model</h2>
<p>Keep all <strong>36 LM decoder layers</strong> and reduce every LM FFN from <strong>11,008 to 4,352 channels</strong>. Audio and vision encoder architecture and channel widths stay intact. Importance scores are calibrated on <strong>128 training clips</strong> per task, with a common selection rule. The pruning geometry and recovery hyperparameters are shared across the three settings; each has separately trained weights.</p>
<div class="workflow"><div class="flow-step"><div class="step-no">Teacher preparation</div><h4>Task SFT on the original model</h4><p>Train fresh rank-16 / alpha-32 decoder q/v LoRA for five epochs on the task’s labelled training split.</p><p>Select by validation accuracy/F1; merge the adapter before structural pruning. Keep this unpruned task checkpoint as the fixed teacher.</p><span class="flow-end">Original task teacher →</span></div>
<div class="flow-step step2"><div class="step-no">Recovery round 1</div><h4>Activation-prune and recover the LM</h4><p>Physically remove low-importance FFN channels. Train fresh rank-16 / alpha-32 adapters on decoder attention and FFN linears for five epochs.</p><p>Use SFT + class distillation; keep the original encoders frozen. Merge the best-validation adapter into the compact LM.</p><span class="flow-end">Recovered 1.925B LM →</span></div>
<div class="flow-step step3"><div class="step-no">Recovery round 2</div><h4>Adapt LM and original encoders</h4><p>Train fresh rank-64 / alpha-128 LoRA on decoder linears and encoder attention, FFN and output/merger projections.</p><p>Five epochs of SFT + class KD + audio/vision feature KD. Dense base weights remain frozen during adapter training.</p><span class="flow-end">Chosen final model</span></div></div>
<div class="callout"><strong>One structural pruning operation, two recovery rounds.</strong> The second round adapts the original encoders and the recovered LM without further pruning. The same original unpruned task teacher supplies class and feature targets. Merge final adapters into their dense matrices for deployment; this preserves the compact architecture and parameter counts.</div>
''' + table(['Setting','Teacher preparation','LM recovery','Final recovery'],[
        ['LoRA scope','Decoder q/v','Decoder q/k/v/o and gate/up/down','Decoder linears + encoder attention/FFN/output and vision merger'],
        ['Rank / alpha','16 / 32','16 / 32','64 / 128'],
        ['Epochs','5','5','5'],
        ['Peak learning rate','1e-4','1e-4','LM 1e-4; audio/vision 1e-5'],
        ['Objective','Answer-letter + EOS SFT','0.25 SFT + 0.75 class KD','Same class objective + 0.1 audio feature KD + 0.1 vision feature KD'],
        ['KD temperature','—','2','2'],
        ['Shared optimizer','AdamW; effective batch 4','Cosine LR; 3% warmup','Dropout 0; weight decay 0; gradient clipping 1.0'],
    ]) + '''<details><summary>Activation scoring, adapter coverage and distillation details</summary><div>
<p><strong>Activation importance.</strong> Hook the actual input to each FFN down projection after the gated activation. Compute mean absolute activation over valid tokens within each calibration clip, then an L2 reduction across clips. Retain the highest-scoring channels, preserve their order and slice the coupled gate/up rows and down columns. Residual width, attention, vocabulary and native media interfaces remain intact.</p>
<p>The score follows the reviewed Minitron / NVIDIA Model Optimizer implementation. This is an adaptation of established structured pruning, not a reproduction of their full architecture search or pretraining procedure. <a href="#src-methods">[4]</a></p>
<span class="formula">L<sub>LM recovery</sub> = 0.25 L<sub>SFT</sub> + 0.75 T² KL(p<sub>teacher</sub> || p<sub>student</sub>), T = 2<br>L<sub>final recovery</sub> = L<sub>LM recovery</sub> + 0.1 L<sub>audio cosine</sub> + 0.1 L<sub>vision cosine</sub></span>
<p>Class KD is computed over the allowed class letters, rather than the full vocabulary. Feature KD matches packed valid-token native encoder outputs with mean cosine distance. Teacher logits and features are cached only for training examples.</p>
<p>Final encoder adapters cover audio attention q/k/v/out, fc1/fc2 and the output projection; vision attention q/k/v/proj, gate/up/down FFNs and both linear layers of the merger. The base weights stay frozen. A two-update GPU probe verifies nonzero adapter gradients and saved updates in all three components, followed by an adapter-reload check.</p>
<p>Final recovery uses one RTX 3090 per task, BF16, gradient checkpointing, microbatch 2 × accumulation 2, and deterministic seed 42. The chosen configuration has 172.09M trainable adapter parameters before merging.</p></div></details></section>''')
    result_rows = []
    for s,label in [('original','Original task SFT · unpruned reference'),('lm_only','LM recovery · intermediate checkpoint'),('final','Chosen model · LM + encoder adaptation')]:
        result_rows.append([f'<strong>{label}</strong>',*[num(pct(primary[s]['runs'][t]['test_primary'])) for t in TASKS]])
    parameter_rows = []
    for s,label in [('original','Original classification Thinker'),('final','Chosen classification Thinker')]:
        p = primary[s]['runs']['mintrec2']['parameters']
        parameter_rows.append([label,*[num(billions(p[k])) for k in ('language','audio','vision','total')],'<span class="tag good">MET</span>' if s=='final' else '<span class="tag context">LM ABOVE LIMIT</span>'])
    body.append('''<section class="section" id="results"><div class="section-kicker">05 · Official test results</div>
<h2>One shared method across all three settings</h2>
<p>All scores use complete official test sets and validation-selected checkpoints. Each task is trained separately with seed 42. MIntRec columns report accuracy; MER reports weighted F1.</p>''' + table(['Configuration','MIntRec-2 ACC','MIntRec-20 ACC','MER weighted F1'],result_rows,cls='results-table',row_classes=['','','emphasis']) + '''
<p>The final model reaches <strong>92.13% binary accuracy</strong>, <strong>78.43% 20-class accuracy</strong> and <strong>88.63% MER weighted F1</strong>. Both logit and greedy evaluation yield identical primary scores on all three tasks, with no invalid greedy answers. The comparison below uses this final model consistently.</p>
<h3>Component sizes after adapter merging</h3>''' + table(['Model','LM','Audio','Vision','Total Thinker','Year-two limits'],parameter_rows,cls='results-table') + f'''<p class="caption">The LM parameter reduction is <strong>{reduction:.2f}%</strong>; the complete classification Thinker reduction is <strong>{total_reduction:.2f}%</strong>. These are measured dense parameter counts, not quantized storage equivalents. Both recovery rounds share the same dense architecture.</p>
<details><summary>Exact parameter counts and both evaluation modes</summary><div>''' + table(['Model','Language','Audio','Vision','Total'],[[label,*[num(f'{primary[s]["runs"]["mintrec2"]["parameters"][k]:,}') for k in ('language','audio','vision','total')]] for s,label in [('original','Original'),('final','Chosen model')]]) + table(['Task','Official test clips','Logit primary metric','Greedy primary metric','Invalid greedy answers'],[[NAMES[t],num(final[t]['test_count']),num(pct(final[t]['test']['logit'][final[t]['primary_metric']])),num(pct(final[t]['test']['greedy'][final[t]['primary_metric']])),num(final[t]['test']['greedy']['invalid_count'])] for t in TASKS]) + '''</div></details>
<h3>Supporting comparison: five pruning approaches</h3>
<p>The controlled comparison covers uniform layer removal, weight-magnitude FFN selection, block-influence layer selection, activation-based FFN selection and a layer/FFN hybrid. Every arm uses five epochs, fresh rank-16 decoder-linear LoRA and original encoders. Within each method, SFT and KD start independently from the same pruned base. Each cell shows <strong>SFT-only → SFT + class KD</strong>; this comparison uses KD weight 0.5 and temperature 2.</p>''')
    titles = {'depth_uniform':'Uniform whole-layer pruning','ffn_magnitude':'Weight-magnitude FFN pruning','depth_bi':'Block-influence layer pruning','ffn_activation':'Activation-based FFN pruning','depth_ffn_activation':'Layer + activation-FFN pruning'}
    method_rows = []
    for m in data['methods']:
        runs = m['runs'];p=runs['mintrec2']['SFT']['parameters']
        method_rows.append([f'<strong>{titles[m["id"]]}</strong><span class="sub">{m["decoder_layers"]} decoder layers · FFN {m["lm_ffn_width"]:,}</span>',num(billions(p['language'])),*[num(f'{100*runs[t]["SFT"]["test_primary"]:.2f} → {100*runs[t]["KD"]["test_primary"]:.2f}') for t in TASKS]])
    body.append(table(['Pruning family','LM size','MIntRec-2 ACC','MIntRec-20 ACC','MER weighted F1'],method_rows,cls='method-table',row_classes=['emphasis' if m['id']=='ffn_activation' else '' for m in data['methods']]) + '''
<p class="caption">Uniform depth pruning keeps 17 evenly spaced layers, including endpoints. Block influence keeps 17 layers with the greatest mean input/output cosine change; the hybrid keeps 28 layers with 6,144 activation-selected FFN channels. Calibrated methods use 128 training clips. These controls use a common comparison recipe; the chosen method’s two-round KD075 recovery and encoder adaptation are specified separately.</p>
<p>Activation-based FFN pruning provides the strongest intent results among these controls while preserving full decoder depth. Additional experimental controls cover adapter rank, learning rate, class/feature distillation, calibration size, encoder compression and joint versus staged recovery. They support method selection without combining per-task winners from different architectures.</p>
<p class="small muted">The chosen final recipe is reported for one training seed. Repeated-seed evaluation remains useful, particularly because the binary result exceeds the hard threshold by 0.13 percentage points.</p></section>''')
    body.append(literature_comparison_section(data,primary))
    controls=''.join(f'<button type="button" data-stage="{s}" aria-pressed="{str(s=="final").lower()}">{STAGE_LABELS[s]}</button>' for s in STAGES)
    loss_html=''.join(f'<figure class="loss-figure" data-stage-panel="{s}"><div class="figure-title">{STAGE_LABELS[s]} · supervised loss</div><div class="chart-scroll">{losses[s]}</div><figcaption>Faint lines show per-update minibatch SFT losses; the solid line is a trailing 51-update mean. Validation loss uses the full development set. The dashed line marks the checkpoint selected by validation performance. The loss axis is logarithmic.</figcaption></figure>' for s in STAGES)
    curve_notes=[]
    explanations={'mintrec2':'The final epoch has the highest validation accuracy within the five-epoch recovery budget.',
                  'mintrec20':'Validation accuracy peaks before the last epoch; the selected checkpoint is retained for testing.',
                  'mer6':'Validation weighted F1 peaks near epoch 4.5; the final epoch is slightly lower.'}
    for t in TASKS:
        r=final[t];b=r['best_validation']
        curve_notes.append(f'<div class="curve-note"><strong>{NAMES[t]} · final recovery</strong>Validation {"weighted F1" if t=="mer6" else "accuracy"} starts at {pct(r["initial_validation"]["score"])} and reaches {pct(b["score"])} at epoch {b["epoch"]:.1f}. {explanations[t]}</div>')
    epoch_rows=[[NAMES[t],*[num(f'{primary[s]["runs"][t]["best_validation"]["epoch"]:.1f}') for s in STAGES]] for t in TASKS]
    runtime_rows=[]
    for s in STAGES:
        for t in TASKS:
            r=primary[s]['runs'][t];e=r['execution']
            runtime_rows.append([STAGE_LABELS[s],NAMES[t],num(f'{e["elapsed_seconds"]/3600:.2f} h'),num(e['world_size']),num(f'{e["trainable_parameters"]/1e6:.2f}M'),num(pct(r['best_validation']['score']))])
    body.append('''<section class="section" id="curves"><div class="section-kicker">07 · Training evidence</div>
<h2>Measured learning curves and checkpoint selection</h2>
<p>Figures are redrawn from saved full-run metrics. Each horizontal axis covers its own five-epoch training stage; stages have different starting weights. Hollow validation markers identify the selected checkpoint. These are validation curves, with official test results reported separately.</p>
<figure><div class="figure-title">Full-validation performance across the method’s training stages</div><div class="chart-scroll">''' + validation + '''</div><figcaption>Original task SFT, LM recovery and final LM/encoder adaptation on all three settings. Selection uses logit accuracy for MIntRec and logit weighted F1 for MER. The final recovery’s validation schedule uses logit evaluation; greedy evaluation is performed on the selected official-test checkpoint.</figcaption></figure>
<h3>Supervised training and validation loss</h3><div class="stage-controls" role="group" aria-label="Select the training stage for supervised loss plots">''' + controls + '</div>' + loss_html + '''
<p class="caption">Curves compare the supervised component, not differently weighted composite SFT/KD/feature objectives. SFT-only logs store this term under <code>loss</code>; distillation logs expose <code>sft_loss</code>. Raw JSONL/CSV metrics, resumable checkpoints and local curve images are retained for every selected run.</p>
<div class="curve-notes">''' + ''.join(curve_notes) + '''</div>
<h3>Selected checkpoint epochs</h3>''' + table(['Task','Teacher SFT','LM recovery','Final recovery'],epoch_rows) + '''
<details><summary>Measured training execution and validation scores</summary><div>
<p>Elapsed times include each training stage’s validation and checkpoint work. Teacher SFT uses four GPUs per task; recovery uses one GPU. These measurements describe training cost, not inference speed. Calibration, teacher-cache construction and official tests are separate operations.</p>''' + table(['Stage','Task','Elapsed','GPUs','Trainable LoRA','Selected validation metric'],runtime_rows) + '''
<p>Effective batch size is 4 for all stages. Four-GPU teacher runs use one example per worker; one-GPU recoveries use microbatch 2 × accumulation 2. Final recovery enables gradient checkpointing. Exact settings and provenance are embedded in the result download.</p></div></details></section>''')
    todo_rows=[
        ['<span class="tag open">PENDING</span>','Confirm consortium reporting scope','Document how the separate consortium emotion F1 ≥0.93 indicator relates to our assigned efficiency and top-five benchmark deliverables.'],
        ['<span class="tag open">PENDING</span>','Quantify robustness','Repeat the fixed final recovery recipe across seeds; report score variation and the binary-threshold margin without changing data splits.'],
        ['<span class="tag open">PENDING</span>','Measure deployment efficiency','Export merged models and measure end-to-end latency, throughput, peak inference memory and storage under the same hardware/software/media protocol.'],
        ['<span class="tag open">PENDING</span>','Publish the model and code','Package the chosen task checkpoints, verify model/data licensing, supply a model card, exact component counts and reproducible training/evaluation commands.'],
        ['<span class="tag open">PENDING</span>','Prepare project evidence','Consolidate benchmark tables, coordinator clarifications, parameter accounting and the requested S/W registration, 공개SW and technical-document materials.'],
    ]
    body.append('''<section class="section" id="next"><div class="section-kicker">08 · Remaining deliverables</div>
<h2>Deployment evidence, robustness and release</h2>
<p>The selected checkpoints establish the assigned size, binary-accuracy and published-comparison results. Remaining work turns those experimental results into the project’s documented, reproducible software deliverable.</p>''' + table(['Status','Work item','Completion criterion'],todo_rows,cls='todo-list') + '''
<h3>Future work</h3>
<p><strong>Year-four compression.</strong> The stricter LM ≤1.5B and individual encoder ≤0.5B budgets require additional compression. Encoder FFN/layer pruning and class/feature distillation provide available building blocks; size-compatible prototypes require their own performance validation before being presented as year-four solutions.</p>
<p><strong>Additional modalities and text.</strong> The MER AVT pipeline provides an optional explicit-transcript setting. Sensor and missing-modality integration follow the consortium’s 3- and 5-modality roadmap and remain separate from this report’s AV/AVT benchmark protocols.</p></section>''')
    provenance_rows=[[STAGE_LABELS[s]+'<br>'+NAMES[t],f'<code>{esc(primary[s]["runs"][t]["sources"]["training_report"])}</code><br><code>{esc(primary[s]["runs"][t]["sources"]["official_test_report"])}</code>'] for s in STAGES for t in TASKS]
    body.append('''<section class="section" id="sources"><div class="section-kicker">Sources and reproducibility</div>
<h2>Raw evidence and a reproducible report build</h2>
<p class="small">This HTML embeds its figures, styles, controls and result data. It works offline; citation links are optional external references. Repository paths identify local evidence and are not rendering dependencies.</p>
<ol class="source-list"><li id="src-plan"><strong>Formal project documents.</strong> <em>IITP087_ICT R&amp;D 수행과제 연차보고서_1차년도.pdf</em>: physical PDF page 8 (overall objective/duration), page 44 / printed 33 (annual KPI table), page 55 / printed 44 (year-two WP4 targets). Amended <em>연구개발계획서(1세부)_최종협약변경반영.hwp</em>: section 2-3, WP4.1/WP4.2 and annual performance indicators. HWP layout pages are not inferred from extracted paragraphs.</li>
<li id="src-slides"><strong>Lab presentation.</strong> <em>스마트홈 워크샵 발표자료(KAIST 이성주 교수님).pptx</em>: slides 2–4 (foundations), slide 6 (2B / one model release / rank 5), slide 7 (efficiency and evaluation), slide 8 (MER-SEMI comparison proposal).</li>
<li id="src-email"><strong>Coordinator correspondence supplied by the user.</strong> 21 September 2026: Qwen2.5-Omni baseline, MIntRec binary ≥92%, supplementary 20-class comparison, MER2023-SEMI F1/top-five and release/modality roadmap. 26 October 2026: encoder size applies per modality—year two ≤1B and year four ≤0.5B. Records: <code>experiments/HANDOFF.md</code> and <code>experiments/DECISIONS.md</code>.</li>
<li id="src-methods"><strong>Pruning references and implementation audit.</strong> <a href="https://arxiv.org/abs/2407.14679" target="_blank" rel="noopener">Minitron</a> and <a href="https://arxiv.org/abs/2403.03853" target="_blank" rel="noopener">ShortGPT</a>. Reviewed code: Minitron <code>fed740c</code>, NVIDIA Model Optimizer <code>d7cbff3</code>, ShortGPT <code>9a2ee8d</code>. Local importance-score fixture audit: <code>docs/internal/CALIBRATED_PRUNING.md</code> and <code>artifacts/implementation/calibrated_pruning_20261019/reference_audit.json</code>.</li>
<li><strong>Published comparisons.</strong> Original MIntRec Table 4; ARL Table 1 and appendix Table 9a; Emotion-LLaMA Table 3. Full citations and inclusion rules appear under <a href="#baselines">Published comparisons</a>. Numerical catalog: <code>artifacts/analysis/baselines_20261103/report_comparison_tables.json</code>.</li>
<li><strong>Chosen method and verified cross-task results.</strong> <code>docs/internal/MER_NATIVE_ENCODERS.md</code>, <code>docs/internal/MINTREC_SHARED_RECIPE.md</code> and <code>artifacts/full/mintrec_shared_recipe_20261106/comparison.json</code>. The comparison recomputes metrics from full official predictions, checks test IDs/labels and selected-checkpoint identities, verifies shared recovery settings and component counts, and ranks against the fixed published catalog.</li>
<li><strong>Measured curves and data.</strong> Completed run reports, <code>metrics.jsonl</code>, <code>resolved_config.json</code> and best-validation checkpoints. The report extractor checks complete official split sizes, score equality, parameter sums and matching validation records. Source hashes accompany the downloadable data.</li></ol>
<details><summary>Selected-run provenance and build commands</summary><div>''' + table(['Stage / task','Training report / official test report'],provenance_rows) + f'''
<p>Workspace: <code>{esc(ROOT)}</code></p>
<pre>python -B scripts/extract_project_report.py
python -B scripts/build_project_report.py</pre>
<p>Verified data: <code>artifacts/analysis/report_20261107/report_data.json</code><br>Source index: <code>{esc(data['source_index']['path'])}</code><br>Source-index SHA-256: <code>{esc(data['source_index']['sha256'])}</code></p>
<p>The experiment inventory tracks {inv['tracked']} logical training entries: {inv['completed_training']} completed training runs, {inv['completed_official_tests']} completed official tests and {inv['cancelled']} cancelled training entries. Preparation jobs, smoke probes and exports are outside these scientific-run counts; some completed training runs are intermediate sources without an official test.</p></div></details></section>''')
    download = dict(data)
    download['chosen_method'] = 'Activation-based LM FFN pruning to 4352 channels; original encoders; decoder KD075 recovery then broad rank64 decoder/encoder SFT + class/feature KD.'
    embedded = json.dumps(download,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    js = r'''
(function(){
 const buttons=Array.from(document.querySelectorAll('[data-stage]'));
 const panels=Array.from(document.querySelectorAll('[data-stage-panel]'));
 function selectStage(stage){buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.stage===stage)));panels.forEach(p=>p.hidden=p.dataset.stagePanel!==stage);}
 buttons.forEach(b=>b.addEventListener('click',()=>selectStage(b.dataset.stage)));selectStage('final');
 document.getElementById('print-report').addEventListener('click',()=>window.print());
 document.getElementById('download-data').addEventListener('click',()=>{
   const data=JSON.parse(document.getElementById('report-data').textContent);
   const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
   const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='iitp_smarthome_report_data_20261107.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 const links=Array.from(document.querySelectorAll('.sidebar a'));
 if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);if(visible.length){const id=visible[0].target.id;links.forEach(a=>a.setAttribute('aria-current',String(a.getAttribute('href')==='#'+id)));}},{rootMargin:'-5% 0px -72% 0px',threshold:0});document.querySelectorAll('section[id]').forEach(s=>observer.observe(s));}
 let prePrint=[];window.addEventListener('beforeprint',()=>{prePrint=Array.from(document.querySelectorAll('details')).map(d=>[d,d.open]);prePrint.forEach(([d])=>d.open=true);});window.addEventListener('afterprint',()=>{prePrint.forEach(([d,open])=>d.open=open);});
})();
'''
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="IITP Smarthome year-two efficiency: chosen Qwen2.5-Omni pruning and distillation method, verified official results, benchmark comparisons and measured learning curves."><title>IITP Smarthome · Year 2 Efficiency Report</title><style>''' + CSS + '''</style></head><body><div class="page"><div class="topline"><div class="brand">IITP Smarthome / Efficiency report</div><div class="actions"><button class="button" type="button" id="download-data">Download result data</button><button class="button" type="button" id="print-report">Print / Save PDF</button></div></div><div class="layout"><nav class="sidebar" aria-label="Report contents"><div class="label">Report contents</div><ol>''' + navhtml + '''</ol><div class="side-note">Year-two focus<br>English technical report<br>Self-contained / offline<br><br>ACC: intent accuracy<br>F1: MER weighted F1<br>KD: teacher distillation</div></nav><main class="report" id="main">''' + ''.join(body) + '''<footer class="footer">IITP Smarthome · Efficiency workstream · 7 November 2026.<br>Official results use the specified task protocols and separately trained checkpoints. Ranks refer to the three named score sources. Deployment measurements, robustness and release preparation are separate deliverables.</footer></main></div></div><script id="report-data" type="application/json">''' + embedded + '</script><script>' + js + '</script></body></html>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ART/'report_data.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/iitp_smarthome_progress.html')
    p.add_argument('--baselines',type=Path,default=BASELINES)
    args=p.parse_args()
    data=json.loads(args.data.read_text())
    data['literature_comparisons']=json.loads(args.baselines.read_text())
    data['literature_comparisons']['our_candidate']='final'
    data['literature_comparisons']['rules']=[
        'Our scores come from the verified chosen-model official logit and greedy tests, selected on validation data.'
        if rule.startswith('Our scores come from') else rule
        for rule in data['literature_comparisons']['rules']]
    content=build(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(content)
    manifest={'output':str(args.output),'bytes':args.output.stat().st_size,'sha256':hashlib.sha256(content.encode()).hexdigest(),
              'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'primary_runs':9,'method_comparison_runs':30,
              'external_render_dependencies':0,'chosen_comparison_report':'artifacts/full/mintrec_shared_recipe_20261106/comparison.json',
              'literature_comparison_sha256':hashlib.sha256(args.baselines.read_bytes()).hexdigest(),
              'literature_score_sources':len(data['literature_comparisons']['sources']),
              'literature_external_configurations':{t:len(v['external']) for t,v in data['literature_comparisons']['tasks'].items()}}
    (ART/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
