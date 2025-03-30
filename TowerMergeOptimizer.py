import streamlit as st
import pandas as pd
import pulp

# --- CSVデータの読み込み ---
@st.cache_data
def load_data():
    df_cosmic_oculus = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Cosmic:Oculus.csv")
    df_crystal_pylon = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Crystal:Pylon.csv")
    df_volt = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Volt.csv")
    df_archmage = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_ArchMage.csv")
    df_flak = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Flak.csv")
    df_mage_lightning = pd.read_csv("https://raw.githubusercontent.com/Flood-wd/TowerMergeOptimizer/main/Data_Mage:Lightning.csv")
    return df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning

df_cosmic_oculus, df_crystal_pylon, df_volt, df_archmage, df_flak, df_mage_lightning = load_data()

# --- 最適化関数（mergeのみ） ---
def optimize_merge(tower_type, initial_level, target_level, max_num, ember_limit=0):
    """
    タワーの統合のみを最適化し、必要な統合用タワーの組み合わせを計算する関数。
    
    :param tower_type: "Cosmic/Oculus", "Crystal/Pylon", "Volt", "ArchMage", "Flak", "Mage/Lightning"
    :param initial_level: 現在のタワーレベル
    :param target_level: 目標タワーレベル
    :param max_num: 統合に利用するタワーの最大使用個数
    :param ember_limit: エンバー上限（0の場合は上限なし）
    :return: (統合するタワーの組み合わせの文字列, 統合前後のリソース比較表 DataFrame)
    """
    # 使用するタワーデータ（直接レベルアップは考慮せず、目標レベルの必要資源との差分でmergeする）
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
        raise ValueError("tower_type must be one of 'Cosmic/Oculus', 'Crystal/Pylon', 'Volt', 'ArchMage', 'Flak', 'Mage/Lightning'")
    
    # 目標レベルの累積リソースとXPを取得
    target_rubble = df_target[df_target['level'] == target_level]['culmative_rubble'].values[0]
    initial_rubble = df_target[df_target['level'] == initial_level]['culmative_rubble'].values[0]
    required_rubble = target_rubble - initial_rubble
    
    target_xp = df_target[df_target['level'] == target_level]['XP_culmative'].values[0]
    initial_xp = df_target[df_target['level'] == initial_level]['XP_culmative'].values[0]
    required_xp = target_xp - initial_xp
    
    # merge候補：FlakおよびMage/Lightningタワー（レベル10以上のもの）
    df_merge_candidates = pd.concat([df_flak.assign(type="Flak"), df_mage_lightning.assign(type="Mage/Lightning")])
    df_merge_candidates = df_merge_candidates[df_merge_candidates['level'] >= 10]
    
    # 最適化モデルの作成（整数計画問題）
    model = pulp.LpProblem("TowerMergeOptimization", pulp.LpMinimize)
    
    # 変数定義（各候補タワーの個数）
    tower_levels = df_merge_candidates[['level', 'type']].values.tolist()
    tower_vars = {tuple(row): pulp.LpVariable(f"x_{row[1]}_{row[0]}", lowBound=0, cat='Integer') for row in tower_levels}
    
    # 目的関数: 統合にかかる総時間の最小化
    model += pulp.lpSum([
        tower_vars[(level, t_type)] *
        df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['time_culmative(days)'].values[0]
        for level, t_type in tower_levels
    ])
    
    # 制約1: rubble量が必要量を満たす（95%の効率を考慮）
    model += pulp.lpSum([
        tower_vars[(level, t_type)] *
        df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['culmative_rubble'].values[0]
        for level, t_type in tower_levels
    ]) * 0.95 >= required_rubble
    
    # 制約2: XPが必要量を満たす
    model += pulp.lpSum([
        tower_vars[(level, t_type)] *
        df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['XP_culmative'].values[0]
        for level, t_type in tower_levels
    ]) >= required_xp
    
    # 制約3: 使用するタワーは最大 max_num 個
    model += pulp.lpSum([tower_vars[(level, t_type)] for level, t_type in tower_levels]) <= max_num

    # 制約4: エンバー上限（ember_limit > 0の場合のみ追加）
    if ember_limit > 0:
        model += pulp.lpSum([
            tower_vars[(level, t_type)] *
            df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['elementalEmber_culmative'].values[0]
            for level, t_type in tower_levels
        ]) <= ember_limit

    # 最適化の実行
    model.solve(pulp.PULP_CBC_CMD(mip=True))
    
    if pulp.LpStatus[model.status] != "Optimal":
        return "エンバーが足りません！", pd.DataFrame()
    
    selected_towers = {
        (level, t_type): int(tower_vars[(level, t_type)].varValue)
        for level, t_type in tower_levels if tower_vars[(level, t_type)].varValue > 0
    }
    
    tower_output = "\n".join([f"レベル{level}の{t_type}タワーを{count}個" for (level, t_type), count in selected_towers.items()])
    
    resource_comparison = pd.DataFrame({
        "Resource": ["ElectrumBar", "ElementalEmber", "CosmicCharge", "Time (days)"],
        "Before Merge": [
            df_target[df_target['level'] == target_level]['electrumBar_culmative'].values[0] - 
            df_target[df_target['level'] == initial_level]['electrumBar_culmative'].values[0],
            df_target[df_target['level'] == target_level]['elementalEmber_culmative'].values[0] - 
            df_target[df_target['level'] == initial_level]['elementalEmber_culmative'].values[0],
            df_target[df_target['level'] == target_level]['cosmicCharge_culmative'].values[0] - 
            df_target[df_target['level'] == initial_level]['cosmicCharge_culmative'].values[0],
            df_target[df_target['level'] == target_level]['time_culmative(days)'].values[0] - 
            df_target[df_target['level'] == initial_level]['time_culmative(days)'].values[0]
        ],
        "After Merge": [
            0,
            sum(selected_towers[(level, t_type)] * 
                df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]
                ['elementalEmber_culmative'].values[0]
                for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * 
                df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]
                ['cosmicCharge_culmative'].values[0]
                for (level, t_type) in selected_towers),
            sum(selected_towers[(level, t_type)] * 
                df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]
                ['time_culmative(days)'].values[0]
                for (level, t_type) in selected_towers)
        ]
    })
    
    return tower_output, resource_comparison

# --- Streamlit UI ---
st.title("Tower Merge Optimizer")

# ユーザー入力
tower_type = st.selectbox("タワーの種類を選択", ["Cosmic/Oculus", "Crystal/Pylon", "Volt", "ArchMage", "Flak", "Mage/Lightning"])
initial_level = st.number_input("現在のレベル", min_value=10, step=1)
target_level = st.number_input("目標レベル", min_value=11, step=1)
max_num = st.number_input("最大使用タワー数（統合に利用するタワーの上限）", min_value=1, max_value=50, value=12)

# エンバー上限の設定（チェックボックスで上限なしがデフォルト）
set_ember_limit = st.checkbox("エンバー上限を設定する", value=False)
if set_ember_limit:
    ember_limit = st.number_input("エンバー上限", min_value=0.0, value=100.0, step=0.1)
else:
    ember_limit = 0.0  # 0なら上限なし

if st.button("最適化を実行"):
    tower_output, resource_comparison = optimize_merge(tower_type, initial_level, target_level, max_num, ember_limit)
    
    st.subheader("最適なタワーの組み合わせ")
    st.text(tower_output)
    
    if tower_output == "エンバーが足りません！":
        st.error("エンバーが足りません！")
    else:
        st.subheader("リソース比較")
        st.dataframe(resource_comparison)
