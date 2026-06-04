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
                tempo = float(parts[2])
            except ValueError:
                continue
            if ptype == 'CPU':
                cpu_times.append(tempo * S2MS)
            elif ptype == 'IO':
                io_times.append(tempo * S2MS)
    return cpu_times, io_times


# ── 1. Build df_rodadas (standard deviation between 5 run means) ──
records_rodadas = []
for sched, info in SCHEDULERS.items():
    for procs in PROC_COUNTS:
        run_cpu_means, run_io_means = [], []
        for run in RUNS:
            fname = info['pattern'].format(run=run, procs=procs)
            fpath = os.path.join(BASE, info['dir'], fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fpath}")
                continue
            cpu_t, io_t = parse_file(fpath)
            if cpu_t:
                run_cpu_means.append(np.mean(cpu_t))
            if io_t:
                run_io_means.append(np.mean(io_t))
        records_rodadas.append({
            'scheduler': sched,
            'procs': procs,
            'cpu_mean': np.mean(run_cpu_means) if run_cpu_means else np.nan,
            'cpu_std':  np.std(run_cpu_means, ddof=1) if len(run_cpu_means) > 1 else np.nan,
            'io_mean':  np.mean(run_io_means) if run_io_means else np.nan,
            'io_std':   np.std(run_io_means, ddof=1) if len(run_io_means) > 1 else np.nan,
        })
df_rodadas = pd.DataFrame(records_rodadas)

# ── 2. Build df_processos (standard deviation across all processes combined) ──
records_processos = []
for sched, info in SCHEDULERS.items():
    for procs in PROC_COUNTS:
        cpu_all, io_all = [], []
        for run in RUNS:
            fname = info['pattern'].format(run=run, procs=procs)
            fpath = os.path.join(BASE, info['dir'], fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fpath}")
                continue
            cpu_t, io_t = parse_file(fpath)
            cpu_all.extend(cpu_t)
            io_all.extend(io_t)
        records_processos.append({
            'scheduler': sched,
            'procs': procs,
            'cpu_mean': np.mean(cpu_all) if cpu_all else np.nan,
            'cpu_std':  np.std(cpu_all, ddof=1) if len(cpu_all) > 1 else np.nan,
            'io_mean':  np.mean(io_all) if io_all else np.nan,
            'io_std':   np.std(io_all, ddof=1) if len(io_all) > 1 else np.nan,
        })
df_processos = pd.DataFrame(records_processos)

# Print comparison of methods and save CSVs
print("=== METODOLOGIA 1: DESVIO ENTRE RODADAS (Stability) ===")
print(df_rodadas.to_string(index=False))
df_rodadas.to_csv(os.path.join(BASE, 'resumo_resultados_rodadas.csv'), index=False)

print("\n=== METODOLOGIA 2: DESVIO ENTRE PROCESSOS INDIVIDUAIS (Fairness) ===")
print(df_processos.to_string(index=False))
df_processos.to_csv(os.path.join(BASE, 'resumo_resultados_processos.csv'), index=False)

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


def bar_chart(df, mean_col, std_col, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Calculate maximum value + error to define a dynamic offset
    max_val = 0
    for sched in SCHED_ORDER:
        sub = df[df['scheduler'] == sched].set_index('procs')
        for p in PROC_COUNTS:
            if p in sub.index:
                val = sub.loc[p, mean_col] / scale
                err = sub.loc[p, std_col] / scale
                if not np.isnan(val):
                    val_err = val + (err if not np.isnan(err) else 0)
                    if val_err > max_val:
                        max_val = val_err
    
    # Define a padding of 1.5% of the maximum value with a small minimum
    padding = max(max_val * 0.015, 0.05)

    for sched in SCHED_ORDER:
        sub = df[df['scheduler'] == sched].set_index('procs')
        vals = [sub.loc[p, mean_col] / scale if p in sub.index else np.nan for p in PROC_COUNTS]
        errs = [sub.loc[p, std_col] / scale if p in sub.index else np.nan for p in PROC_COUNTS]
        xpos = X + OFFSETS[sched] * W
        bars = ax.bar(xpos, vals, W, label=sched,
                      color=COLORS[sched], edgecolor='white', linewidth=0.6)
        ax.errorbar(xpos, vals, yerr=errs, fmt='none',
                    ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2)
        for bar, v, e in zip(bars, vals, errs):
            if not np.isnan(v):
                err_val = e if not np.isnan(e) else 0
                y_pos = v + err_val + padding
                fmt = f'{v:.1f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        fmt, ha='center', va='bottom', fontsize=7)
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


def line_chart(df, title, filename, scale_cpu=1.0, scale_io=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mean_col, std_col, t, scale, ylabel in [
        (axes[0], 'cpu_mean', 'cpu_std', 'CPU-bound - Tempo medio de retorno', scale_cpu, 'Tempo medio de retorno (ms)'),
        (axes[1], 'io_mean',  'io_std',  'IO-bound - Tempo medio de retorno', scale_io, 'Tempo medio de retorno (s)'),
    ]:
        for sched in SCHED_ORDER:
            sub = df[df['scheduler'] == sched].sort_values('procs')
            ax.errorbar(sub['procs'], sub[mean_col] / scale, yerr=sub[std_col] / scale,
                        label=sched, color=COLORS[sched],
                        marker='o', linewidth=2, markersize=6,
                        capsize=4, capthick=1.2, elinewidth=1.2)
        ax.set_xlabel('Numero de processos', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.set_xticks(PROC_COUNTS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.35)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ── GENERATE PLOTS ──

# Option 1: Standard deviation between 5 run means (Rodadas)
bar_chart(df_rodadas, 'cpu_mean', 'cpu_std',
          'Tempo medio de retorno (ms)',
          'Processos CPU-bound - Tempo medio (Desvio entre Rodadas)',
          'grafico_cpu_media_rodadas.png')

bar_chart(df_rodadas, 'io_mean', 'io_std',
          'Tempo medio de retorno (s)',
          'Processos IO-bound - Tempo medio (Desvio entre Rodadas)',
          'grafico_io_media_rodadas.png',
          scale=1000.0)

line_chart(df_rodadas,
           'Comparacao dos Escalonadores (Desvio entre Rodadas)',
           'grafico_linha_comparativo_rodadas.png',
           scale_io=1000.0)

# Option 2: Standard deviation across individual processes (Processos)
bar_chart(df_processos, 'cpu_mean', 'cpu_std',
          'Tempo medio de retorno (ms)',
          'Processos CPU-bound - Tempo medio (Desvio entre Processos)',
          'grafico_cpu_media_processos.png')

bar_chart(df_processos, 'io_mean', 'io_std',
          'Tempo medio de retorno (s)',
          'Processos IO-bound - Tempo medio (Desvio entre Processos)',
          'grafico_io_media_processos.png',
          scale=1000.0)

line_chart(df_processos,
           'Comparacao dos Escalonadores (Desvio entre Processos)',
           'grafico_linha_comparativo_processos.png',
           scale_io=1000.0)

print("\nConcluido! Todos os arquivos foram gerados e salvos com sucesso.")
