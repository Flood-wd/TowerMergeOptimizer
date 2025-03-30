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

# --- merge（統合）最適化関数 ---
def merge_optimization(required_rubble, required_xp, max_num, ember_limit, df_merge_candidates):
    """
    merge候補（FlakとMage/Lightning）を用いて、必要なrubbleとXPをカバーする最適な組み合わせを
    最小の時間で求める。
    
    :param required_rubble: mergeで補うべき追加rubble量
    :param required_xp: mergeで補うべき追加XP量
    :param max_num: 最大使用タワー数
    :param ember_limit: エンバー上限（0なら上限なし）
    :param df_merge_candidates: マージ候補のデータ（FlakとMage/Lightning）
    :return: (merging_plan_text, total_merge_time)
             解が得られなければ ("エンバーが足りません！", None) を返す。
    """
    model = pulp.LpProblem("MergeOptimization", pulp.LpMinimize)
    
    tower_levels = df_merge_candidates[['level', 'type']].values.tolist()
    tower_vars = {tuple(row): pulp.LpVariable(f"z_{row[1]}_{row[0]}", lowBound=0, cat='Integer') for row in tower_levels}
    
    # 目的関数: mergeタワーの総時間の最小化
    model += pulp.lpSum([
        tower_vars[(level, t_type)] *
        df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['time_culmative(days)'].values[0]
        for level, t_type in tower_levels
    ])
    
    # 制約1: rubble量が必要量を満たす (95%の効率)
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
    
    # 制約3: 使用タワー数はmax_num以内
    model += pulp.lpSum([tower_vars[(level, t_type)] for level, t_type in tower_levels]) <= max_num

    # 制約4: エンバー上限（ember_limit > 0の場合のみ）
    if ember_limit > 0:
        model += pulp.lpSum([
            tower_vars[(level, t_type)] *
            df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['elementalEmber_culmative'].values[0]
            for level, t_type in tower_levels
        ]) <= ember_limit
    
    model.solve(pulp.PULP_CBC_CMD(mip=True))
    
    if pulp.LpStatus[model.status] != "Optimal":
        return "エンバーが足りません！", None
    
    selected_towers = {
        (level, t_type): int(tower_vars[(level, t_type)].varValue)
        for level, t_type in tower_levels if tower_vars[(level, t_type)].varValue > 0
    }
    
    merging_plan = "\n".join([f"レベル{level}の{t_type}タワーを{count}個" for (level, t_type), count in selected_towers.items()])
    total_merge_time = sum(
        selected_towers[(level, t_type)] *
        df_merge_candidates[(df_merge_candidates['level'] == level) & (df_merge_candidates['type'] == t_type)]['time_culmative(days)'].values[0]
        for (level, t_type) in selected_towers
    )
    
    return merging_plan, total_merge_time

# --- 直接レベルアップ＋統合＋再度直接レベルアップを組み合わせた最適化関数 ---
def optimize_levelup_combined(tower_type, initial_level, target_level, max_num, ember_limit):
    """
    直接レベルアップとmerge（統合）を組み合わせた最適化を実施する。
    プランは以下の2段階または3段階の組み合わせを検討する：
      ① 初期レベルから中間レベル m まで直接レベルアップし、
         その後 merge を用いて m から n まで統合し、さらに直接レベルアップで n から目標レベルに到達
         （m < nの場合）
      ② あるいは、純粋に直接レベルアップのみ（m = n = target_level）
    すべての (m, n) 組み合わせ（initial_level ≤ m ≤ n ≤ target_level）について総所要時間を算出し、
    最も時間が短いプランを採用します。
    
    :return: (plan_text, resource_comparison DataFrame)
    """
    # メインタワーデータ（直接レベルアップ対象）
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
    
    # 関連する直接レベルアップの時間と資源（df_target）を利用
    # df_targetにおける各レベルの情報を辞書で簡単にアクセスできるようにする
    target_info = {row['level']: row for _, row in df_target.iterrows()}
    
    # マージ候補：FlakおよびMage/Lightning
    df_merge_candidates = pd.concat([df_flak.assign(type="Flak"), df_mage_lightning.assign(type="Mage/Lightning")])
    df_merge_candidates = df_merge_candidates[df_merge_candidates['level'] >= 10]
    
    best_total_time = float("inf")
    best_plan = None
    best_m = None
    best_n = None
    best_merge_plan = None
    best_direct_time1 = None
    best_direct_time2 = None
    
    # m: 初期レベルからの直接レベルアップで到達する中間レベル
    # n: その後、mergeで到達するレベル（n==mならmerge不要）
    for m in range(initial_level, target_level + 1):
        # 直接レベルアップ部分1：initial -> m
        time_direct1 = target_info[m]['time_culmative(days)'] - target_info[initial_level]['time_culmative(days)']
        for n in range(m, target_level + 1):
            if n == m:
                # 直接レベルアップのみの場合（merge不要）
                merge_time = 0
                merge_plan = "なし"
            else:
                # mergeで m -> n を行う
                required_rubble_merge = target_info[n]['culmative_rubble'] - target_info[m]['culmative_rubble']
                required_xp_merge = target_info[n]['XP_culmative'] - target_info[m]['XP_culmative']
                merge_plan, merge_time = merge_optimization(required_rubble_merge, required_xp_merge, max_num, ember_limit, df_merge_candidates)
                if merge_plan == "エンバーが足りません！":
                    continue  # この (m, n) ではmergeが成立しない
            # 直接レベルアップ部分2：n -> target_level
            time_direct2 = target_info[target_level]['time_culmative(days)'] - target_info[n]['time_culmative(days)']
            
            total_time = time_direct1 + merge_time + time_direct2
            
            if total_time < best_total_time:
                best_total_time = total_time
                best_m = m
                best_n = n
                best_merge_plan = merge_plan
                best_direct_time1 = time_direct1
                best_direct_time2 = time_direct2
                best_plan = (
                    f"【プラン候補】\n"
                    f"① 直接レベルアップ：初期レベル {initial_level} から中間レベル {m} まで → 時間: {time_direct1:.2f} days\n"
                    + (f"② 統合による補填：中間レベル {m} から {n} へ merge を実施 → 所要時間: {merge_time:.2f} days\n統合内容:\n{merge_plan}\n"
                    if n > m else "② 統合による補填：不要（直接レベルアップのみ）\n")
                    + f"③ 直接レベルアップ：中間レベル {n} から目標レベル {target_level} まで → 時間: {time_direct2:.2f} days\n"
                    + f"総所要時間: {total_time:.2f} days"
                )
    
    if best_plan is None:
        return "エンバー上限の条件内でのプランが見つかりません！", pd.DataFrame()
    
    # 統合前後のリソース比較（main tower の場合）
    resource_comparison = pd.DataFrame({
        "Resource": ["Rubble", "XP", "Time (days)"],
        "Direct (初期→目標)": [
            target_info[target_level]['culmative_rubble'] - target_info[initial_level]['culmative_rubble'],
            target_info[target_level]['XP_culmative'] - target_info[initial_level]['XP_culmative'],
            target_info[target_level]['time_culmative(days)'] - target_info[initial_level]['time_culmative(days)']
        ],
        "本プラン (Direct + Merge + Direct)": [
            (target_info[m]['culmative_rubble'] - target_info[initial_level]['culmative_rubble']) +
            (target_info[target_level]['culmative_rubble'] - target_info[best_n]['culmative_rubble']),
            (target_info[m]['XP_culmative'] - target_info[initial_level]['XP_culmative']) +
            (target_info[target_level]['XP_culmative'] - target_info[best_n]['XP_culmative']),
            best_total_time
        ]
    })
    
    return best_plan, resource_comparison

# --- Streamlit UI ---
st.title("Tower Merge & Level-Up Optimizer")

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
    plan_text, resource_comparison = optimize_levelup_combined(tower_type, initial_level, target_level, max_num, ember_limit)
    
    st.subheader("最適なプラン")
    st.text(plan_text)
    
    if resource_comparison.empty:
        st.error("条件内でのプランが見つかりませんでした。")
    else:
        st.subheader("リソース比較")
        st.dataframe(resource_comparison)
