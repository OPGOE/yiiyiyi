# 先强制检查依赖安装（调试用，部署后可保留）
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    import joblib
    import os
    import requests
    from io import StringIO
    st.success("✅ 所有依赖库加载成功！")
except ImportError as e:
    st.error(f"❌ 缺少依赖库：{str(e)}")
    st.error("请确保requirements.txt包含所有依赖并重启应用！")
    st.stop()

# 设置页面配置
st.set_page_config(
    page_title="医疗费用预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 1. 加载CSV文件（本地+远程双兜底） ----------------------
@st.cache_data
def load_data():
    """加载CSV，优先本地，失败则远程读取GitHub Raw"""
    # 配置信息（确认仓库名/分支名正确）
    local_csv = "insurance-chinese.csv"
    github_raw_url = "https://raw.githubusercontent.com/OPGOE/yiliao/main/insurance-chinese.csv"
    encodings = ["utf-8-sig", "gbk", "utf-8", "gb2312"]

    # 本地读取逻辑
    if os.path.exists(local_csv):
        for enc in encodings:
            try:
                df = pd.read_csv(local_csv, encoding=enc, on_bad_lines="skip")
                df.columns = df.columns.str.strip().str.replace(" ", "")
                required_cols = ["年龄", "性别", "子女数量", "是否吸烟", "区域", "医疗费用"]
                if all(col in df.columns for col in required_cols):
                    st.success(f"✅ 本地读取CSV成功（编码：{enc}）")
                    X = df[["年龄", "性别", "子女数量", "是否吸烟", "区域"]]
                    y = df["医疗费用"]
                    return X, y, df
            except:
                continue

    # 远程读取逻辑
    st.info("本地读取失败，尝试远程读取GitHub数据...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(github_raw_url, headers=headers, timeout=15)
        resp.raise_for_status()
        for enc in encodings:
            try:
                resp.encoding = enc
                df = pd.read_csv(StringIO(resp.text), on_bad_lines="skip")
                df.columns = df.columns.str.strip().str.replace(" ", "")
                required_cols = ["年龄", "性别", "子女数量", "是否吸烟", "区域", "医疗费用"]
                if all(col in df.columns for col in required_cols):
                    st.success("✅ 远程读取CSV成功！")
                    X = df[["年龄", "性别", "子女数量", "是否吸烟", "区域"]]
                    y = df["医疗费用"]
                    return X, y, df
            except:
                continue
        st.error("❌ 远程CSV编码解析失败！")
        st.stop()
    except Exception as e:
        st.error(f"❌ 远程读取失败：{str(e)}")
        st.stop()

# ---------------------- 2. 模型训练（极简版，减少报错） ----------------------
def train_model(X, y):
    """简化模型训练逻辑，降低报错概率"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cat_features = ["性别", "是否吸烟", "区域"]
        num_features = ["年龄", "子女数量"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features)
            ]
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=50, random_state=42))  # 减少树数量，加快训练
        ])

        model.fit(X_train, y_train)
        joblib.dump(model, "model.pkl")
        y_pred = model.predict(X_test)
        return model, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    except Exception as e:
        st.error(f"❌ 模型训练失败：{str(e)}")
        st.stop()

# ---------------------- 3. 加载模型 ----------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        try:
            return joblib.load("model.pkl")
        except:
            X, y, _ = load_data()
            model, _, _ = train_model(X, y)
            return model
    else:
        X, y, _ = load_data()
        model, _, _ = train_model(X, y)
        return model

# ---------------------- 4. 页面逻辑（极简版） ----------------------
def main():
    st.sidebar.title("🧭 导航")
    page = st.sidebar.radio("", ["简介", "预测医疗费用"], index=1)

    if page == "简介":
        st.title("🏥 医疗费用预测系统")
        st.markdown("基于机器学习的医疗费用预测工具")
    else:
        st.title("🏥 医疗费用预测系统")
        st.markdown("---")
        
        # 核心加载步骤
        try:
            X, y, df = load_data()
            model = load_model()
        except:
            st.error("初始化失败，请检查CSV文件和依赖！")
            return

        # 输入表单
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("年龄", 0, 100, 30)
            gender = st.radio("性别", ["男性", "女性"], horizontal=True)
            children = st.number_input("子女数量", 0, 10, 0)
        with col2:
            smoker = st.radio("是否吸烟", ["否", "是"], horizontal=True)
            region = st.selectbox("区域", df["区域"].unique())
            bmi = st.number_input("BMI指数", 10.0, 50.0, 25.0)

        # 预测按钮
        if st.button("🚀 预测医疗费用", type="primary"):
            input_data = pd.DataFrame({
                "年龄": [age], "性别": [gender], "子女数量": [children],
                "是否吸烟": [smoker], "区域": [region]
            })
            try:
                pred = model.predict(input_data)[0]
                st.success(f"💰 预计医疗费用：${pred:,.2f}")
            except:
                st.error("预测失败，请检查输入数据！")

if __name__ == "__main__":
    main()
