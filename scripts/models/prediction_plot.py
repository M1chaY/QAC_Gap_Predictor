import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

# 加载 qac_results.csv 并绘制 xlogP vs Gap 散点图
results_df = pd.read_csv(DATA_DIR / "qac_results.csv")

# 处理 gap 值：如果 gap 为空则使用 predicted_gap，都为空则删去
gap_values = pd.to_numeric(results_df['gap'], errors='coerce')
predicted_gap_values = pd.to_numeric(results_df['predicted_gap'], errors='coerce')
results_df['gap_final'] = gap_values.combine_first(predicted_gap_values)
results_df['xlogp'] = pd.to_numeric(results_df['xlogp'], errors='coerce')
results_df = results_df.dropna(subset=['xlogp', 'gap_final'])

# 筛选 xlogP <= 5 的样本中 Gap 最小的前三个
top3_gap_indices = results_df.loc[results_df['xlogp'] <= 5].nsmallest(3, 'gap_final').index

special_condition = results_df.index.isin(top3_gap_indices)

# 绘制散点图 (4:3 比例)
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制其他所有点（圆形，浅蓝色）
ax.scatter(results_df.loc[~special_condition, 'xlogp'], 
           results_df.loc[~special_condition, 'gap_final'],
           c="#3C6AFF", alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
           marker='o')

# 绘制特殊条件的点（五角星，粉红色）
ax.scatter(results_df.loc[special_condition, 'xlogp'], 
           results_df.loc[special_condition, 'gap_final'],
           c="#FF1D1D", alpha=0.9, s=120, edgecolors='black', linewidth=0.5,
           marker='*')

ax.set_xlabel('XlogP')
ax.set_ylabel('HOMO-LUMO Gap (eV)')
ax.set_title('QAC : XlogP vs Gap')

# 在 xlogP=5 位置添加浅灰色虚线
ax.axvline(x=5, color='#8A8A8A', linestyle='--', linewidth=1)

ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(DATA_DIR / "qac_plot.png", dpi=300)
plt.show()

print(f"Total samples: {len(results_df)}")
print(f"Special condition (xlogP<=5, Top 3 smallest Gap): {special_condition.sum()}")
print(f"Others: {(~special_condition).sum()}")

# 输出五角星标记的化合物 SMILES
print("\n" + "=" * 70)
print("Special Compounds (xlogP<=5, Top 3 smallest Gap):")
print("=" * 70)
special_df = results_df.loc[special_condition, ['SMILES', 'xlogp', 'gap_final']].reset_index(drop=True)
for idx, row in special_df.iterrows():
    print(f"{idx+1}. SMILES: {row['SMILES']}")
    print(f"   xlogP: {row['xlogp']:.2f}, Gap: {row['gap_final']:.2f} eV")
