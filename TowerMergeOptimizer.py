import streamlit as st
import pandas as pd
import pulp

# --- CSVデータの読み込み ---
@st.cache_data
def load_data():
    df_cosmic_oculus = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Cosmic/Oculus.csv")
    df_crystal_pylon = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Crystal/Pylon.csv")
    df_volt = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Volt.csv")
    df_archmage = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_ArchMage.csv")
    df_flak = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Flak.csv")
    df_mage_lightning = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Mage/Lightning.csv")
    return df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning

df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning = load_data()

def optimize_merge(tower_type, initial_level, target_level, max_num):
    tower_data_map = {
        "Cosmic/Oculus": df_cosmic_oculus,
        "Crystal/Pylon": df_crystal_pylon,
        "Volt": df_volt,
        "ArchMage": df_archmage,
        "Flak": df_flak,
        "Mage/Lightning": df_mage_lightning
    }
    
    if tower_type not in tower_data_map:
        raise ValueError("tower_type must be one of: " + ", ".join(tower_data_map.keys()))
    
    df_target = tower_data_map[tower_type]
    target_rubble = df_target[df_target['level'] == target_level]['culmative_rubble'].values[0]
    initial_rubble = df_target[df_target['level'] == initial_level]['culmative_rubble'].values[0]
    required_rubble = target_rubble - initial_rubble
    
    target_xp = df_target[df_target['level'] == target_level]['XP_culmative'].values[0]
    initial_xp = df_target[df_target['level'] == initial_level]['XP_culmative'].values[0]
    required_xp = target_xp - initial_xp
    
    df_merge_candidates = pd.concat([df_flak.assign(type="Flak"), df_mage_lightning.assign(type="Mage/Lightning")])
    df_merge_candidates = df_merge_candidates[df_merge_candidates['level'] >= 10]
    
    model = pulp.LpProblem("TowerMergeOptimization", pulp.LpMinimize)
    
    tower_levels = df_merge_candidates[['level', 'type']].values.tolist()
    tower_vars = {tuple(row): pulp.LpVariable(f"x_{row[1]}_{row[0]}", lowBound=0, cat='Integer') for row in tower_levels}
    
    model += pulp.lpSum([tower_vars[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['time_culmative(days)'].values[0] for level, t_type in tower_levels])
    
    model += pulp.lpSum([tower_vars[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['culmative_rubble'].values[0] for level, t_type in tower_levels]) * 0.95 >= required_rubble
    model += pulp.lpSum([tower_vars[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['XP_culmative'].values[0] for level, t_type in tower_levels]) >= required_xp
    model += pulp.lpSum([tower_vars[(level, t_type)] for level, t_type in tower_levels]) <= max_num
    
    model.solve(pulp.PULP_CBC_CMD(mip=True))
    
    selected_towers = {(level, t_type): int(tower_vars[(level, t_type)].varValue) for level, t_type in tower_levels if tower_vars[(level, t_type)].varValue > 0}
    
    tower_output = "\n".join([f"レベル{level}の{t_type}タワーを{count}個" for (level, t_type), count in selected_towers.items()])
    
    resource_comparison = pd.DataFrame({
        "Resource": ["ElectrumBar", "ElementalEmber", "CosmicCharge", "Time (days)"],
        "Before Merge": [
            df_target[df_target['level'] == target_level]['electrumBar_culmative'].values[0] - df_target[df_target['level'] == initial_level]['electrumBar_culmative'].values[0],
            df_target[df_target['level'] == target_level]['elementalEmber_culmative'].values[0] - df_target[df_target['level'] == initial_level]['elementalEmber_culmative'].values[0],
            df_target[df_target['level'] == target_level]['cosmicCharge_culmative'].values[0] - df_target[df_target['level'] == initial_level]['cosmicCharge_culmative'].values[0],
            df_target[df_target['level'] == target_level]['time_culmative(days)'].values[0] - df_target[df_target['level'] == initial_level]['time_culmative(days)'].values[0]
        ],
        "After Merge": [
            0,
            sum(selected_towers[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['elementalEmber_culmative'].values[0] for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['cosmicCharge_culmative'].values[0] for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['time_culmative(days)'].values[0] for (level, t_type) in selected_towers)
        ]
    })
    
    return tower_output, resource_comparison

st.title("タワーマージ最適化ツール")
tower_type = st.selectbox("タワーの種類を選択", list(load_data()[0].keys()))
initial_level = st.number_input("現在のレベル", min_value=10, step=1)
target_level = st.number_input("目標レベル", min_value=11, step=1)
max_num = st.number_input("統合に使用するタワーの最大数", min_value=1, step=1, value=12)

if st.button("最適化を実行"):
    tower_output, resource_comparison = optimize_merge(tower_type, initial_level, target_level, max_num)
    st.subheader("最適なタワーの組み合わせ")
    st.text(tower_output)
    st.subheader("リソース比較")
    st.dataframe(resource_comparison)
