import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

BASE = os.path.dirname(os.path.abspath(__file__))

SCHEDULERS = {
    'Padrao': {
        'dir': 'MINIX-DEFAULT',
        'pattern': 'resultados{run}_minix_{procs}.txt',
    },
    'FCFS': {
        'dir': 'FCFS',
        'pattern': 'resultados{run}_FCFS_{procs}.txt',
    },
    'RR': {
        'dir': 'RR',
        'pattern': 'resultados{run}_rr_{procs}.txt',
    },
    'MF': {
        'dir': 'MF',
        'pattern': 'resultado{run}_mf_{procs}.txt',
    },
}

PROC_COUNTS = [10, 50, 100, 200]
RUNS = [1, 2, 3, 4, 5]
S2MS = 1000  # seconds → milliseconds


def parse_file(path):
    cpu_times, io_times = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            ptype = parts[0]
            try:
                t1 = float(parts[1])
                t2 = float(parts[2])
            except ValueError:
                continue
            if ptype == 'CPU':
                cpu_times.append(t1 * S2MS)
            elif ptype == 'IO':
                io_times.append(t2 * S2MS)
    return cpu_times, io_times


# Build records: one row per (scheduler, procs, run) with per-run mean
records = []
for sched, info in SCHEDULERS.items():
    for procs in PROC_COUNTS:
        for run in RUNS:
            fname = info['pattern'].format(run=run, procs=procs)
            fpath = os.path.join(BASE, info['dir'], fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fpath}")
                continue
            cpu_t, io_t = parse_file(fpath)
            records.append({
                'scheduler': sched,
                'procs': procs,
                'run': run,
                'cpu_run_mean': np.mean(cpu_t) if cpu_t else np.nan,
                'io_run_mean':  np.mean(io_t)  if io_t  else np.nan,
            })

raw = pd.DataFrame(records)

# Aggregate: mean and std across 5 runs
df = (
    raw.groupby(['scheduler', 'procs'])
    .agg(
        cpu_mean=('cpu_run_mean', 'mean'),
        cpu_std= ('cpu_run_mean', 'std'),
        io_mean= ('io_run_mean',  'mean'),
        io_std=  ('io_run_mean',  'std'),
    )
    .reset_index()
)

print(df.to_string(index=False))
df.to_csv(os.path.join(BASE, 'resumo_resultados.csv'), index=False)

# ── Plot settings ──────────────────────────────────────────────
COLORS = {
    'Padrao': '#2196F3',
    'FCFS':   '#FF9800',
    'RR':     '#4CAF50',
    'MF':     '#E91E63',
}
SCHED_ORDER = ['Padrao', 'FCFS', 'RR', 'MF']
X = np.arange(len(PROC_COUNTS))
W = 0.18
OFFSETS = {'Padrao': -1.5, 'FCFS': -0.5, 'RR': 0.5, 'MF': 1.5}


def bar_chart(mean_col, std_col, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    for sched in SCHED_ORDER:
        sub = df[df['scheduler'] == sched].set_index('procs')
        vals = [sub.loc[p, mean_col] if p in sub.index else np.nan for p in PROC_COUNTS]
        errs = [sub.loc[p, std_col]  if p in sub.index else np.nan for p in PROC_COUNTS]
        xpos = X + OFFSETS[sched] * W
        bars = ax.bar(xpos, vals, W, label=sched,
                      color=COLORS[sched], edgecolor='white', linewidth=0.6)
        ax.errorbar(xpos, vals, yerr=errs, fmt='none',
                    ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(errs) * 0.12 + 20,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(X)
    ax.set_xticklabels([str(p) for p in PROC_COUNTS])
    ax.set_xlabel('Numero de processos', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


bar_chart('cpu_mean', 'cpu_std',
          'Tempo medio de retorno (ms)',
          'Processos CPU-bound - Tempo medio de retorno',
          'grafico_cpu_media.png')

bar_chart('io_mean', 'io_std',
          'Tempo medio de retorno (ms)',
          'Processos IO-bound - Tempo medio de retorno',
          'grafico_io_media.png')

# ── Side-by-side comparison (line chart) ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mean_col, std_col, title in [
    (axes[0], 'cpu_mean', 'cpu_std', 'CPU-bound - Tempo medio de retorno'),
    (axes[1], 'io_mean',  'io_std',  'IO-bound - Tempo medio de retorno'),
]:
    for sched in SCHED_ORDER:
        sub = df[df['scheduler'] == sched].sort_values('procs')
        ax.errorbar(sub['procs'], sub[mean_col], yerr=sub[std_col],
                    label=sched, color=COLORS[sched],
                    marker='o', linewidth=2, markersize=6,
                    capsize=4, capthick=1.2, elinewidth=1.2)
    ax.set_xlabel('Numero de processos', fontsize=10)
    ax.set_ylabel('Tempo medio de retorno (ms)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(PROC_COUNTS)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)

plt.suptitle('Comparacao dos Escalonadores', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'grafico_linha_comparativo.png'), dpi=150)
plt.close()
print("Saved: grafico_linha_comparativo.png")

print("\nConcluido. Arquivos gerados:")
print("  resumo_resultados.csv")
print("  grafico_cpu_media.png")
print("  grafico_io_media.png")
print("  grafico_linha_comparativo.png")
