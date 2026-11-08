#!/usr/bin/env python3
"""Verified coordinator-only handoff with a held-dispatch acceptance gate."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.queueing import StateStore, atomic_json, fingerprint, now, process_identity, same_process
from project_queue_jobs import import_project
from project_queue import DEFAULT_ROOT


def direct_native_children(parents):
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            identity=process_identity(int(path.name))
            if int(fields[1]) in parents and identity and any(
                    str(ROOT/'scripts'/name) in identity['command']
                    for name in ['train.py','evaluate.py','prune_model.py','cache_teacher_logits.py']):
                found.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError):pass
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    if not args.apply:
        state=import_project(args.root)
        print(json.dumps({'jobs':len(state['jobs']),'workers':[{'id':j['id'],'pid':j['pid'],'gpu':j.get('gpu')} for j in state['jobs'] if j['status']=='running']},indent=2))
        return
    if (args.root/'state.json').exists():raise RuntimeError('A persistent queue already exists; use project_queue.py run')
    evidence=ROOT/'artifacts/implementation/project_queue_20261019'
    evidence.mkdir(exist_ok=True)
    paused={};killed=[];original=None;worker_ids=[];enabled=False;started=False
    try:
        # Repeat to capture a follow-up coordinator born just before its parent pauses.
        for _ in range(3):
            state=import_project(args.root)
            added=False
            for path,document in state['documents'].items():
                pid=document.get('launcher_pid')
                if not pid or pid in paused:continue
                identity=process_identity(pid)
                if identity is None:continue
                scripts={str(ROOT/'scripts'/name) for name in ['launch_pruning_sweep.py','launch_pruning_followup.py','launch_recovery_sweep.py']}
                cwd=Path(f'/proc/{pid}/cwd').resolve()
                actual_scripts={str((cwd/arg).resolve()) for arg in identity['command'] if arg.endswith('.py')}
                output_arg=identity['command'][identity['command'].index('--output-root')+1] if '--output-root' in identity['command'] else ''
                actual_output=str((cwd/output_arg).resolve()) if output_arg else ''
                if not scripts.intersection(actual_scripts) or str(Path(path).parent) != actual_output:
                    if document.get('status')=='completed':continue
                    raise ValueError('Coordinator identity mismatch; no process will be killed')
                os.kill(pid,signal.SIGSTOP)
                paused[pid]=identity;added=True
            if not added:break
        state=import_project(args.root)
        known={j['pid'] for j in state['jobs'] if j['status']=='running'}
        unexpected=set(direct_native_children(set(paused)))-known
        if unexpected:raise RuntimeError(f'Coordinator paused during worker publication; retry safely: {unexpected}')
        original=copy.deepcopy(state['documents'])
        worker_ids=[{'id':j['id'],'pid':j['pid'],'identity':j.get('identity'),'gpu':j.get('gpu')} for j in state['jobs'] if j['status']=='running']
        atomic_json(evidence/'handoff_before.json',{'at':now(),'documents':original,'coordinators':paused,'workers':worker_ids})
        state['queue_root']=str(args.root.resolve());state['handoff']={'at':now(),'coordinators_replaced':list(paused),'workers_signalled':0}
        for pid,identity in paused.items():
            if not same_process(pid,identity):raise ValueError('Coordinator identity changed before replacement')
            os.kill(pid,signal.SIGKILL);killed.append(pid)
        for _ in range(100):
            if all(process_identity(pid) is None for pid in killed):break
            time.sleep(.1)
        else:raise RuntimeError('Legacy coordinators did not release locks')
        StateStore(args.root).save(state)
        command = shlex.join([sys.executable, '-B', '-u', str(ROOT/'scripts/project_queue.py'),
                              'run', '--root', str(args.root)])
        command = f'exec {command} > {shlex.quote(str(args.root/"scheduler.log"))} 2>&1'
        subprocess.run(['tmux','new-session','-d','-s','smarthome_project_queue','-c',str(ROOT),command],check=True)
        started=True
        for _ in range(200):
            current=StateStore(args.root).load()
            pid=current.get('scheduler_pid')
            if pid and same_process(pid,current.get('scheduler_identity')) and current.get('status')=='running':break
            time.sleep(.1)
        else:raise RuntimeError('New scheduler did not become ready; see scheduler.log')
        preserved=[]
        for worker in worker_ids:
            if same_process(worker['pid'],worker['identity']):
                preserved.append(worker['pid'])
            else:
                observed=next(j for j in current['jobs'] if j['id']==worker['id'])
                if observed['status']!='completed':
                    # A worker may have just completed; let report reconciliation finish.
                    time.sleep(1)
                    latest=StateStore(args.root).load()
                    observed=next(j for j in latest['jobs'] if j['id']==worker['id'])
                    if observed['status']!='completed':raise RuntimeError('A worker disappeared without validated completion')
        # Only now allow new workers. Existing workers were observed throughout.
        (args.root/'dispatch.enabled').write_text('Verified coordinator-only handoff '+now()+'\n')
        enabled=True
        atomic_json(evidence/'handoff.json',{'passed':True,'at':now(),'old_coordinators':killed,
                    'scheduler_pid':pid,'workers_before':worker_ids,'preserved_live_worker_pids':preserved,
                    'workers_signalled':0,'dispatch_enabled_after_verification':True})
        print(json.dumps({'handoff':'passed','scheduler_pid':pid,'preserved_worker_pids':preserved,'workers_signalled':0}))
    except BaseException:
        if not killed:
            for pid in paused:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
        elif not enabled and original is not None:
            # Before dispatch, rollback is possible without any new native workers.
            if started:
                try:
                    current=StateStore(args.root).load();pid=current.get('scheduler_pid')
                    if pid:os.kill(pid,signal.SIGTERM)
                    for _ in range(200):
                        if not pid or process_identity(pid) is None:break
                        time.sleep(.1)
                except (OSError,ValueError):pass
            for path,document in original.items():
                for job in document['jobs']:
                    if job.get('status')=='running':
                        identity=process_identity(job['pid'])
                        job['process_identity']=identity or {'start_ticks':'already-exited','command':job['command']}
                atomic_json(path,document)
            import shlex
            for index,(old_pid,identity) in enumerate(paused.items()):
                command=list(identity['command'])
                if '--resume' not in command:command.append('--resume')
                if any('launch_pruning_' in part for part in command) and '--adopt-running' not in command:
                    command.append('--adopt-running')
                subprocess.run(['tmux','new-session','-d','-s',f'smarthome_rollback_{index}',
                                '-c',str(ROOT),'exec '+shlex.join(command)],check=True)
            atomic_json(evidence/'rollback.json',{'at':now(),'coordinators_restored':True,'workers_signalled':0})
        raise


if __name__=='__main__':main()
