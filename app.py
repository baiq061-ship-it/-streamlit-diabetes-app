# app.py
import streamlit as st
import joblib, json, pandas as pd, numpy as np, os

# 0. 必须是第一个 Streamlit 命令
st.set_page_config(page_title="糖尿病预后预测", layout="centered")

# 1. 加载模型与元数据
RESULT_DIR = "diabetes_analysis_results"
META_FILE  = os.path.join(RESULT_DIR, "path_index.json")

@st.cache_resource
def load_artifacts():
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(meta["best_model_path"])
    feats = meta["feature_info"]["feature_names"]
    return model, feats

model, FEATURE_ORDER = load_artifacts()

# 2. 页面元素
st.title("🩺 院内心脏骤停患者一年神经功能预测")
st.markdown("> 上传 CSV 或手动输入特征，即可实时获得预测概率")

# 3. 乱码→简体 映射表（根据报错信息填写）
RENAME_MAP = {
    "CA鐥呭洜": "CA病因",
    "ROSC鍚庣櫧铔嬬櫧": "ROSC后白蛋白",
    "蹇冭偤澶嶈嫃鏃堕棿": "心肺复苏时间",
    "鑲句笅鑵虹礌鎬婚噺": "肾上腺素总量"
}

# 4. 侧边栏：选择输入方式
input_mode = st.sidebar.radio("输入方式", ["手动输入", "批量上传 CSV"])

if input_mode == "手动输入":
    with st.form("single"):
        vals = {}
        for f in FEATURE_ORDER:
            vals[f] = st.number_input(f, value=0.0, format="%.4f")
        submitted = st.form_submit_button("预测")
        if submitted:
            vals = {RENAME_MAP.get(k, k): v for k, v in vals.items()}  # 修复乱码
            X = pd.DataFrame([vals])[FEATURE_ORDER]
            proba = model.predict_proba(X)[0, 1]
            st.success(f"神经功能良好概率：{proba:.1%}")
            st.progress(proba)

else:  # 批量上传
    uploaded = st.file_uploader("上传 CSV（必须包含以下列）", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df = df.rename(columns=RENAME_MAP)  # 修复乱码
        miss = set(FEATURE_ORDER) - set(df.columns)
        if miss:
            st.error(f"缺少列：{miss}")
        else:
            df["预测概率"] = model.predict_proba(df[FEATURE_ORDER])[:, 1]
            st.write(df)
            csv = df.to_csv(index=False)
            st.download_button("下载带概率文件", csv, "predictions.csv")

# 5. 底部说明
with st.expander("模型说明"):
    st.markdown("""
    - 本模型由 Optuna 自动调参生成，AUC 见训练日志  
    - 仅用于科研演示，不可直接用于临床决策  
    - 特征顺序必须与训练时完全一致
    """)
