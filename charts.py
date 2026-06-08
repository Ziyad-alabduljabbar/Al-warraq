import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

OUTPUT_DIR = r"C:\Users\mobil\Desktop\‏‏Al warraq\report_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = r"C:\Users\mobil\Desktop\‏‏Al warraq"
eval_df     = pd.read_csv(os.path.join(BASE, 'evaluation_results.csv'))
speed_df    = pd.read_csv(os.path.join(BASE, 'speed_results.csv'))
conc_df     = pd.read_csv(os.path.join(BASE, 'concurrent_results.csv'))
chatbot_df  = pd.read_csv(os.path.join(BASE, 'chatbot_results.csv'))
spell_df    = pd.read_csv(os.path.join(BASE, 'spell_evaluation.csv'))

for df in [eval_df, speed_df, conc_df, chatbot_df, spell_df]:
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

TOP_K = int(open(os.path.join(BASE, 'evaluate.py')).read().split('TOP_K')[1].split('=')[1].split('\n')[0].strip())
N     = len(eval_df)

COLORS = {
    'std': '#2E5EA8', 'int': '#1A8065',
    'fav': '#C4440A', 'spl': '#8A857D', 'gold': '#B8860B',
}
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})

def save(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {name}.png")


# 01. Hit Rate by Mode
hr_std  = eval_df['HR_Standard'].mean()
hr_int  = eval_df['HR_Interests'].mean()
hr_fav  = eval_df['HR_Favorites'].mean()
hr_spl  = eval_df['HR_Spell'].mean()

fig, ax = plt.subplots(figsize=(7, 4))
modes  = ['Standard', 'Interests', 'Favorites', 'Spell']
values = [hr_std, hr_int, hr_fav, hr_spl]
colors = [COLORS['std'], COLORS['int'], COLORS['fav'], COLORS['spl']]
bars = ax.bar(modes, values, color=colors, width=0.5, zorder=3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(values) * 1.25)
ax.set_ylabel('Hit Rate (%)', fontsize=11)
ax.set_title(f'Hit Rate by Retrieval Mode  (n={N}, TOP_K={TOP_K})', fontsize=13, fontweight='bold', pad=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('01_hit_rate_by_mode')


# 02. Three Metrics Comparison
hr   = [eval_df['HR_Standard'].mean(),   eval_df['HR_Interests'].mean(),   eval_df['HR_Favorites'].mean(),   eval_df['HR_Spell'].mean()]
ndcg = [eval_df['NDCG_Standard'].mean(), eval_df['NDCG_Interests'].mean(), eval_df['NDCG_Favorites'].mean(), eval_df['NDCG_Spell'].mean()]
p1   = [eval_df['P1_Standard'].mean(),   eval_df['P1_Interests'].mean(),   eval_df['P1_Favorites'].mean(),   eval_df['P1_Spell'].mean()]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(4)
w = 0.25
ax.bar(x - w, hr,   w, label='Hit Rate',          color=COLORS['std'], zorder=3)
ax.bar(x,     ndcg, w, label=f'NDCG@{TOP_K}',    color=COLORS['int'], zorder=3)
ax.bar(x + w, p1,   w, label='P@1',               color=COLORS['fav'], zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(['Standard', 'Interests', 'Favorites', 'Spell'])
ax.set_ylim(0, max(p1) * 1.2)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title(f'Three Metrics Comparison by Mode  (TOP_K={TOP_K})', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('02_metrics_comparison')


# 03. Improvement vs Standard
# ── 3. Improvement vs Standard ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3))
labels = ['Interests', 'Favorites', 'Spell']
imp_int = eval_df['HR_Imp_Interests'].mean()
imp_fav = eval_df['HR_Imp_Favorites'].mean()
imp_spl = eval_df['HR_Imp_Spell'].mean()

fig, ax = plt.subplots(figsize=(6, 3))
values = [imp_int, imp_fav, imp_spl]
colors = [COLORS['int'], COLORS['fav'], COLORS['spl']]
bars = ax.barh(labels, values, color=colors, height=0.4, zorder=3)

for bar, val in zip(bars, values):
    sign = '+' if val >= 0 else ''
    if val >= 0:
       
        x_pos = val + 0.3
        ha = 'left'
    else:
        
        x_pos = 0.3
        ha = 'left'
    ax.text(x_pos,
            bar.get_y() + bar.get_height() / 2,
            f'{sign}{val:.1f}pp',
            va='center', ha=ha,
            fontsize=11, fontweight='bold',
            color='#C4440A' if val < 0 else 'black')  

ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlim(min(values) - 3, max(values) + 5)   
ax.set_xlabel('Percentage-point change vs Standard', fontsize=10)
ax.set_title('Improvement vs Standard Baseline', fontsize=13, fontweight='bold', pad=12)
ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('03_improvement_vs_standard')


# 04. Interests Count Impact
grouped = eval_df.groupby('Num_Interests')['HR_Interests'].mean()
counts  = eval_df.groupby('Num_Interests').size()

fig, ax = plt.subplots(figsize=(6, 4))
ni_colors = [COLORS['int'], COLORS['gold'], COLORS['fav']]
x_labels  = [f'{int(n)} Interest{"s" if n>1 else ""}\n(n={counts[n]})' for n in grouped.index]
bars = ax.bar(x_labels, grouped.values, color=ni_colors[:len(grouped)], width=0.4, zorder=3)
for bar, val in zip(bars, grouped.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(grouped.values) * 1.25)
ax.set_ylabel('Hit Rate — Interests Mode (%)', fontsize=11)
ax.set_title('Effect of Number of Interests on Hit Rate', fontsize=13, fontweight='bold', pad=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('04_interests_count_impact')


# 05. Speed Latency
def stats(col):
    return [speed_df[col].mean(), speed_df[col].min(),
            speed_df[col].max(), speed_df[col].quantile(0.95)]

std_vals = stats('Mode1_ms')
int_vals = stats('Mode2_ms')
fav_vals = stats('Mode3_ms')

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(4)
w = 0.25
ax.bar(x - w, std_vals, w, label='Standard',  color=COLORS['std'], zorder=3)
ax.bar(x,     int_vals, w, label='Interests', color=COLORS['int'], zorder=3)
ax.bar(x + w, fav_vals, w, label='Favorites', color=COLORS['fav'], zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(['Avg', 'Min', 'Max', 'P95'])
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title(f'Latency Distribution by Mode  (n={len(speed_df)})', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('05_speed_latency')


# 06. Concurrent Scaling
users = conc_df['Num_Users'].tolist()
avg   = conc_df['Avg_ms'].tolist()
p95   = conc_df['P95_ms'].tolist()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(users, avg, 'o-', color=COLORS['std'], linewidth=2.5, markersize=7, label='Avg latency')
ax.fill_between(users, avg, alpha=0.08, color=COLORS['std'])
ax.plot(users, p95, 's--', color=COLORS['fav'], linewidth=2.5, markersize=7, label='P95 latency')
ax.fill_between(users, p95, alpha=0.05, color=COLORS['fav'])
ax.axhline(100, color='gray', linestyle=':', linewidth=1, label='100ms threshold')
for x_val, y_val in zip(users, avg):
    ax.text(x_val, y_val + 2.5, f'{y_val}ms', ha='center', fontsize=9, color=COLORS['std'])
for x_val, y_val in zip(users, p95):
    ax.text(x_val, y_val + 2.5, f'{y_val}ms', ha='center', fontsize=9, color=COLORS['fav'])
ax.set_xticks(users)
ax.set_xlabel('Concurrent Users', fontsize=11)
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title('Latency Scaling with Concurrent Users', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('06_concurrent_scaling')


# 07. Degradation Factor
deg = conc_df['Degradation_x'].tolist()
deg_colors = [COLORS['int'] if d <= 2 else COLORS['gold'] if d <= 3.5 else COLORS['fav'] for d in deg]

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar([str(int(u))+' user'+('s' if u>1 else '') for u in users],
              deg, color=deg_colors, width=0.4, zorder=3)
for bar, val in zip(bars, deg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val}×', ha='center', fontsize=11, fontweight='bold')
ax.axhline(5, color='red', linestyle='--', linewidth=1, label='Max threshold (5×)')
ax.set_ylim(0, 6)
ax.set_ylabel('Degradation Factor', fontsize=11)
ax.set_title('Degradation Factor by User Level', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('07_degradation_factor')


# 08. Chatbot Latency
cb_ids  = chatbot_df['Test_ID'].tolist()
cb_lats = chatbot_df['Latency_ms'].tolist()
cb_cols = [COLORS['std'], COLORS['int'], COLORS['gold'], COLORS['fav'], COLORS['spl']]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(cb_ids, cb_lats, color=cb_cols, width=0.5, zorder=3)
for bar, val, tid in zip(bars, cb_lats, cb_ids):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}ms', ha='center', fontsize=10, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, max(cb_lats)*0.03,
                '0ms\n(blocked)', ha='center', fontsize=8, color='gray')
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title('Chatbot Response Latency by Scenario', fontsize=13, fontweight='bold', pad=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('08_chatbot_latency')


# 09. Edge Cases
edge_df = eval_df[eval_df['Query_Type'].str.startswith('Edge', na=False)].copy()
edge_labels = [r['Query_Type'].replace('Edge_','') + f'\n({r["Search_Query"][:10]})' 
               for _, r in edge_df.iterrows()]

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(edge_df))
w = 0.25
ax.bar(x - w, edge_df['HR_Standard'].values,  w, label='Standard',  color=COLORS['std'], zorder=3)
ax.bar(x,     edge_df['HR_Interests'].values,  w, label='Interests', color=COLORS['int'], zorder=3)
ax.bar(x + w, edge_df['HR_Favorites'].values,  w, label='Favorites', color=COLORS['fav'], zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(edge_labels, fontsize=9)
ax.set_ylim(0, 110)
ax.set_ylabel('Hit Rate (%)', fontsize=11)
ax.set_title(f'Edge Cases — Hit Rate by Mode  (TOP_K={TOP_K})', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
save('09_edge_cases')

print(f"\nDone — {TOP_K} TOP_K — n={N}")
print(f"Charts saved to: {OUTPUT_DIR}")