# 依赖导入（完全静默成功提示，仅保留错误提示）
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
    # 无任何成功提示输出（删除所有st.success/print）
except ImportError as e:
    st.error(f"❌ 缺少依赖库：{str(e)}")
    st.error("请确保requirements.txt包含所有依赖并重启应用！")
    st.stop()

# 页面基础配置
st.set_page_config(
    page_title="医疗费用预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 1. 加载CSV文件（完全静默成功提示） ----------------------
@st.cache_data
def load_data():
    """加载CSV文件，仅在失败时显示错误，成功无任何提示"""
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
                    X = df[["年龄", "性别", "子女数量", "是否吸烟", "区域"]]
                    y = df["医疗费用"]
                    return X, y, df
            except:
                continue

    # 远程读取逻辑（无任何信息提示）
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
                    X = df[["年龄", "性别", "子女数量", "是否吸烟", "区域"]]
                    y = df["医疗费用"]
                    return X, y, df
            except:
                continue
        st.error("❌ 远程CSV文件格式错误，缺少必要列或编码不兼容！")
        st.stop()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"❌ 远程CSV文件不存在（404），请检查链接：{github_raw_url}")
        else:
            st.error(f"❌ 远程读取失败（HTTP {e.response.status_code}）")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("❌ 连接GitHub超时，请检查网络！")
        st.stop()
    except Exception as e:
        st.error(f"❌ CSV读取失败：{str(e)}")
        st.stop()

# ---------------------- 2. 模型训练（仅失败时提示） ----------------------
def train_model(X, y):
    """训练模型，成功无提示，失败显示错误"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
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
            ("regressor", RandomForestRegressor(n_estimators=50, random_state=42))
        ])

        model.fit(X_train, y_train)
        joblib.dump(model, "model.pkl")
        y_pred = model.predict(X_test)
        return model, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    except Exception as e:
        st.error(f"❌ 模型训练失败：{str(e)}")
        st.stop()

# ---------------------- 3. 加载模型（仅失败时提示） ----------------------
@st.cache_resource
def load_model():
    """加载模型，成功无提示，失败自动重新训练"""
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

# ---------------------- 4. 页面主逻辑 ----------------------
def main():
    # 侧边栏导航
    st.sidebar.title("🧭 导航")
    page = st.sidebar.radio(
        "",
        ["简介", "预测医疗费用"],
        index=1
    )

    # 简介页面
    if page == "简介":
        st.title("🏥 医疗费用预测系统")
        st.markdown("---")
        st.markdown("""
        ## 📋 系统简介
        本系统是基于机器学习的医疗费用预测工具，旨在为保险公司和医疗机构提供准确的费用预测参考。
        
        ### 🎯 主要功能
        - **智能预测**: 基于随机森林算法，准确预测个人年度医疗费用
        - **多因素分析**: 综合考虑年龄、性别、吸烟状况、子女数量、地区等因素
        - **风险评估**: 自动识别高风险因素并提供健康建议
        - **实时计算**: 输入信息后即时获得预测结果
        
        ### 📝 使用说明
        1. 点击左侧导航中的"预测医疗费用"
        2. 填写被保险人的基本信息
        3. 点击"预测医疗费用"按钮
        4. 查看预测结果和风险提示
        
        💡 **提示**: 预测结果仅供参考，实际医疗费用可能因个人健康状况、医疗政策等因素而有所不同。
        """)
    
    # 预测页面
    else:
        st.title("🏥 医疗费用预测系统")
        st.markdown("---")
        st.markdown("基于外部CSV数据的医疗费用预测工具")
        st.markdown("---")
        
        # 核心加载步骤（无成功提示）
        try:
            X, y, df = load_data()
            model = load_model()
        except Exception as e:
            st.error(f"❌ 系统初始化失败：{str(e)}")
            return

        # 输入表单
        st.subheader("📝 被保险人信息")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("年龄", min_value=0, max_value=100, value=30, step=1)
            gender = st.radio("性别", options=["男性", "女性"], horizontal=True)
            children = st.number_input("子女数量", min_value=0, max_value=10, value=0, step=1)
        with col2:
            smoker = st.radio("是否吸烟", options=["否", "是"], horizontal=True)
            region_options = df["区域"].unique().tolist() if len(df["区域"].unique()) > 0 else ["东北", "西北", "东南", "西南"]
            region = st.selectbox("区域", options=region_options)
            bmi = st.number_input("BMI指数", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

        # 预测按钮
        st.markdown("---")
        if st.button("🚀 预测医疗费用", type="primary"):
            input_data = pd.DataFrame({
                "年龄": [age],
                "性别": [gender],
                "子女数量": [children],
                "是否吸烟": [smoker],
                "区域": [region]
            })
            try:
                prediction = model.predict(input_data)[0]
                st.success(f"💰 预计年度医疗费用：${prediction:,.2f}")
                
                # 风险提示
                warnings = []
                if smoker == "是": warnings.append("吸烟会显著增加医疗费用风险")
                if bmi > 30: warnings.append("BMI过高可能增加健康风险")
                if age > 60: warnings.append("年龄较大，医疗费用风险较高")
                if warnings:
                    st.markdown("---")
                    st.subheader("⚠️ 风险提示")
                    for w in warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"❌ 预测失败：{str(e)}")

if __name__ == "__main__":
    main()
