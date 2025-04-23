import streamlit as st
import pandas as pd
import pulp

# ----------------------------------------
# データ読み込み・最適化処理（バックエンド）
# ----------------------------------------

@st.cache_data
def load_data():
    df_cosmic_oculus = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_.xlsx%20-%20Cosmic_Oculus.csv")
    df_crystal_pylon = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_.xlsx%20-%20Crystal_Pylon.csv")
    df_volt = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_.xlsx%20-%20Volt.csv")
    df_archmage = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_.xlsx%20-%20ArchMage.csv")
    df_flak = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_.xlsx%20-%20Flak.csv")
    df_mage_lightning = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_.xlsx%20-%20Mage_lightning.csv")
    return df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning

def optimize_merge(tower_type, initial_level, target_level, max_num, material_type,
                   df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning):
    
    if tower_type == "Cosmic/Oculus":
        df_target = df_cosmic_oculus
    elif tower_type == "Crystal/Pylon":
        df_target = df_crystal_pylon
    elif tower_type == "Volt":
        df_target = df_volt
    elif tower_type == "ArchMage":
        df_target = df_archmage
    elif tower_type == "Flak":
        df_target = df_flak
    elif tower_type == "Mage/Lightning":
        df_target = df_mage_lightning
    else:
        raise ValueError("Invalid tower_type.")

    if material_type == "ElementalEmber":
        df_merge_candidates = pd.concat([
            df_flak.assign(type="Flak"),
            df_mage_lightning.assign(type="Mage/Lightning")
        ])
    elif material_type == "ElectrumBar":
        df_merge_candidates = pd.concat([
            df_crystal_pylon.assign(type="Crystal/Pylon"),
            df_mage_lightning.assign(type="Mage/Lightning")
        ])
    else:
        raise ValueError("Invalid material_type.")

    df_merge_candidates = df_merge_candidates[df_merge_candidates['level'] >= 10]

    target_rubble = df_target[df_target['level'] == target_level]['cumulative_rubble'].values[0]
    initial_rubble = df_target[df_target['level'] == initial_level]['cumulative_rubble'].values[0]
    required_rubble = target_rubble - initial_rubble

    target_xp = df_target[df_target['level'] == target_level]['XP_cumulative'].values[0]
    initial_xp = df_target[df_target['level'] == initial_level]['XP_cumulative'].values[0]
    required_xp = target_xp - initial_xp

    model = pulp.LpProblem("TowerMergeOptimization", pulp.LpMinimize)

    tower_levels = df_merge_candidates[['level', 'type']].values.tolist()
    tower_vars = {tuple(row): pulp.LpVariable(f"x_{row[1]}_{row[0]}", lowBound=0, cat='Integer') for row in tower_levels}

    model += pulp.lpSum([
        tower_vars[(level, t_type)] * df_merge_candidates[
            (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
        ]['time_cumulative(days)'].values[0]
        for level, t_type in tower_levels
    ])

    model += pulp.lpSum([
        tower_vars[(level, t_type)] * df_merge_candidates[
            (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
        ]['cumulative_rubble'].values[0]
        for level, t_type in tower_levels
    ]) * 0.95 - 250 >= required_rubble

    model += pulp.lpSum([
        tower_vars[(level, t_type)] * df_merge_candidates[
            (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
        ]['XP_cumulative'].values[0]
        for level, t_type in tower_levels
    ]) >= required_xp

    model += pulp.lpSum([
        tower_vars[(level, t_type)]
        for level, t_type in tower_levels
    ]) <= max_num

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_towers = {
        (level, t_type): int(tower_vars[(level, t_type)].varValue)
        for level, t_type in tower_levels
        if tower_vars[(level, t_type)].varValue > 0
    }

    tower_output = "\n".join([
        f"レベル{level}の{t_type}タワーを{count}個"
        for (level, t_type), count in selected_towers.items()
    ])

    resource_comparison = pd.DataFrame({
        "Resource": ["ElectrumBar", "ElementalEmber", "CosmicCharge", "Lumber", "Time (days)"],
        "Before Merge": [
            df_target[df_target['level'] == target_level]['electrumBar_cumulative'].values[0] - df_target[df_target['level'] == initial_level]['electrumBar_cumulative'].values[0],
            df_target[df_target['level'] == target_level]['elementalEmber_cumulative'].values[0] - df_target[df_target['level'] == initial_level]['elementalEmber_cumulative'].values[0],
            df_target[df_target['level'] == target_level]['cosmicCharge_cumulative'].values[0] - df_target[df_target['level'] == initial_level]['cosmicCharge_cumulative'].values[0],
            df_target[df_target['level'] == target_level]['Lumber_cumulative'].values[0] - df_target[df_target['level'] == initial_level]['Lumber_cumulative'].values[0],
            df_target[df_target['level'] == target_level]['time_cumulative(days)'].values[0] - df_target[df_target['level'] == initial_level]['time_cumulative(days)'].values[0]
        ],
        "After Merge": [
            sum(selected_towers[(level, t_type)] * df_merge_candidates[
                (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
            ]['electrumBar_cumulative'].values[0] for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * df_merge_candidates[
                (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
            ]['elementalEmber_cumulative'].values[0] for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * df_merge_candidates[
                (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
            ]['cosmicCharge_cumulative'].values[0] for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * df_merge_candidates[
                (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
            ]['Lumber_cumulative'].values[0] for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * df_merge_candidates[
                (df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)
            ]['time_cumulative(days)'].values[0] for (level, t_type) in selected_towers)
        ]
    })

    return tower_output, resource_comparison

# ------------------------------
# Streamlit UI（フロントエンド）
# ------------------------------

st.title("Tower Merge Optimizer")

df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning = load_data()

tower_type = st.selectbox("タワーの種類を選択", [
    "Cosmic/Oculus", "Crystal/Pylon", "Volt", "ArchMage", "Flak", "Mage/Lightning"
])

col1, col2 = st.columns(2)
with col1:
    initial_level = st.number_input("現在のレベル", min_value=10, step=1)
with col2:
    target_level = st.number_input("目標レベル", min_value=11, step=1)

col3, col4 = st.columns(2)
with col3:
    max_num = st.number_input("最大使用タワー数", min_value=1, max_value=50, value=12)
with col4:
    material_type = st.selectbox("素材タイプを選択", ["ElementalEmber", "ElectrumBar"])

if st.button("最適化を実行"):
    tower_output, resource_comparison = optimize_merge(
        tower_type, initial_level, target_level, max_num, material_type,
        df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning
    )

    st.subheader("最適なタワーの組み合わせ")
    st.text(tower_output)

    st.subheader("リソース比較")
    st.dataframe(resource_comparison)

