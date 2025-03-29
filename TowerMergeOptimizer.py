import streamlit as st
import pandas as pd
import pulp

# --- CSVデータの読み込み ---
@st.cache_data
def load_data():
    df_flak = pd.read_csv("/Users/ohmizutakashi/Downloads/Data_Flak.csv")
    df_cosmic = pd.read_csv("/Users/ohmizutakashi/Downloads/Data_Cosmic.csv")
    df_crystal = pd.read_csv("/Users/ohmizutakashi/Downloads/Data_Crystal.csv")
    df_lumber = pd.read_csv("/Users/ohmizutakashi/Downloads/Data_Lumbertower.csv")
    return df_flak, df_cosmic, df_crystal, df_lumber

df_flak, df_cosmic, df_crystal, df_lumber = load_data()

# --- 最適化関数 ---
def optimize_merge(tower_type, initial_level, target_level):
    """
    タワーの統合を最適化し、必要なタワーの組み合わせを計算
    
    :param tower_type: "Cosmic", "Crystal", または "Lumber"
    :param initial_level: 現在のタワーレベル
    :param target_level: 目標タワーレベル
    """
    # 使用するタワーデータを選択
    if tower_type.lower() == "cosmic":
        df_target = df_cosmic
    elif tower_type.lower() == "crystal":
        df_target = df_crystal
    elif tower_type.lower() == "lumber":
        df_target = df_lumber
    else:
        raise ValueError("tower_type must be either 'Cosmic', 'Crystal', or 'Lumber'")
    
    # 目標レベルのrubbleとXPを取得
    target_rubble = df_target[df_target['level'] == target_level]['culmative_rubble'].values[0]
    initial_rubble = df_target[df_target['level'] == initial_level]['culmative_rubble'].values[0]
    required_rubble = target_rubble - initial_rubble
    
    target_xp = df_target[df_target['level'] == target_level]['XP_culmative'].values[0]
    initial_xp = df_target[df_target['level'] == initial_level]['XP_culmative'].values[0]
    required_xp = target_xp - initial_xp
    
    # FlakおよびLumberタワーを統合素材として使用（レベル10以上のものを抽出）
    df_merge_candidates = pd.concat([df_flak.assign(type="Flak"), df_lumber.assign(type="Lumber")])
    df_merge_candidates = df_merge_candidates[df_merge_candidates['level'] >= 10]
    
    # 最適化モデルの作成（整数計画問題）
    model = pulp.LpProblem("TowerMergeOptimization", pulp.LpMinimize)
    
    # 変数定義（各タワーレベルの個数）
    tower_levels = df_merge_candidates[['level', 'type']].values.tolist()
    tower_vars = {tuple(row): pulp.LpVariable(f"x_{row[1]}_{row[0]}", lowBound=0, cat='Integer') for row in tower_levels}
    
    # 目的関数: 統合後の総時間の最小化
    model += pulp.lpSum([tower_vars[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['time_culmative(days)'].values[0] for level, t_type in tower_levels])
    
    # 制約1: rubble量が必要量を満たす
    model += pulp.lpSum([tower_vars[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['culmative_rubble'].values[0] for level, t_type in tower_levels]) * 0.95 >= required_rubble
    
    # 制約2: XPが統合前より多いこと
    model += pulp.lpSum([tower_vars[(level, t_type)] * df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['XP_culmative'].values[0] for level, t_type in tower_levels]) >= required_xp
    
    # 制約3: 使用するタワーは最大12個
    model += pulp.lpSum([tower_vars[(level, t_type)] for level, t_type in tower_levels]) <= max_num
    
    # 最適化の実行
    model.solve(pulp.PULP_CBC_CMD(mip=True))
    
    # 結果の取得
    selected_towers = {(level, t_type): int(tower_vars[(level, t_type)].varValue) for level, t_type in tower_levels if tower_vars[(level, t_type)].varValue > 0}
    
    # タワーの出力フォーマット
    tower_output = "\n".join([f"レベル{level}の{t_type}タワーを{count}個" for (level, t_type), count in selected_towers.items()])
    
    # 統合前後のリソース消費の比較表
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
# --- Streamlit UI ---
st.title("タワーマージ最適化ツール")

# ユーザー入力
tower_type = st.selectbox("タワーの種類を選択", ["Cosmic", "Crystal", "Lumber"])
initial_level = st.number_input("現在のレベル", min_value=10, step=1)
target_level = st.number_input("目標レベル", min_value=11, step=1)

if st.button("最適化を実行"):
    tower_output, resource_comparison = optimize_merge(tower_type, initial_level, target_level)
    
    st.subheader("最適なタワーの組み合わせ")
    st.text(tower_output)
    
    st.subheader("リソース比較")
    st.dataframe(resource_comparison)