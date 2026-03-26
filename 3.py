#1.1读取数据

import pandas as pd
import numpy as np

# 数据集路径
file_path = r"C:\Users\study\Desktop\模块一\online_shoppers_intention.csv"

# 读取数据
df = pd.read_csv(file_path)

# 查看数据基本形状
print("数据集形状：", df.shape)



#1.2查看前几行和字段名
# 查看前5行
display(df.head())

# 查看全部字段名
print("字段名列表：")
print(df.columns.tolist())



#1.3查看数据类型和缺失值
# 查看数据基本信息
df.info()

print("\n每一列的缺失值个数：")
print(df.isnull().sum())



#1.4描述统计和 Revenue 分布
# 数值型字段描述统计
display(df.describe())

# Revenue 分布统计
print("Revenue 分布：")
print(df["Revenue"].value_counts())

print("\nRevenue 占比：")
print(df["Revenue"].value_counts(normalize=True))



#1.5导入绘图库并设置风格
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 全局绘图风格设置
# -----------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文显示
plt.rcParams["axes.unicode_minus"] = False     # 负号正常显示

sns.set_theme(style="whitegrid")

# 马卡龙配色
macaron_colors = [
    "#A8D8EA",  # 浅蓝
    "#AA96DA",  # 浅紫
    "#FCBAD3",  # 浅粉
    "#FFFFD2",  # 浅黄
    "#B8E1DD",  # 浅青
    "#FFD3B6"   # 浅橙
]
#设置字体
# 中文 -> 黑体（SimHei）
# 英文/数字 -> Times New Roman
# =============================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid")




#6.数值型特征相关性热力图
# -----------------------------
# 数值型特征相关性热力图（学术风 + 中英文字体区分）
# -----------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 9))
ax = sns.heatmap(
    corr_matrix,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    square=True,
    center=0,
    cbar_kws={"shrink": 0.8}
)

plt.title("数值型特征相关性热力图", fontsize=16, pad=15, fontname="SimHei")
plt.xticks(rotation=45, ha="right", fontsize=10, fontname="Times New Roman")
plt.yticks(rotation=0, fontsize=10, fontname="Times New Roman")

# colorbar字体
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()



#7.关键特征差异总览图
# -----------------------------
# Revenue分布 + 关键特征分组对比（字体修正版）
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# 全局字体
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid")

# 关键特征
key_features = [
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "PageValues"
]

titles_cn = [
    "浏览商品页数量与购买结果对比",
    "商品页停留时间与购买结果对比",
    "跳出率与购买结果对比",
    "页面价值与购买结果对比"
]

# 配色
bar_colors = ["#A8D8EA", "#FCBAD3"]
box_palette = ["#A8D8EA", "#FCBAD3"]

# 创建画布
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# ========= 图1：Revenue 分布 =========
revenue_counts = df["Revenue"].value_counts()
revenue_labels = ["未购买", "已购买"]

bars = axes[0].bar(
    revenue_labels,
    revenue_counts.values,
    color=bar_colors,
    width=0.6,
    edgecolor="black",
    linewidth=1.0
)

axes[0].set_title("Revenue 类别分布", fontsize=15, pad=12, fontname="SimHei")
axes[0].set_xlabel("购买结果", fontsize=12, fontname="SimHei")
axes[0].set_ylabel("样本数量", fontsize=12, fontname="SimHei")

# 设置刻度字体
for label in axes[0].get_xticklabels():
    label.set_fontname("SimHei")
    label.set_fontsize(11)

for label in axes[0].get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(10)

# 柱顶标注
total = revenue_counts.sum()
for bar, value in zip(bars, revenue_counts.values):
    percent = value / total * 100
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 120,
        f"{value}\n({percent:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontname="Times New Roman"
    )

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# ========= 图2-5：关键特征按Revenue分组的箱线图 =========
for i, col in enumerate(key_features, start=1):
    sns.boxplot(
        data=df,
        x="Revenue",
        y=col,
        ax=axes[i],
        palette=box_palette,
        width=0.6,
        showfliers=False
    )

    # 中文标题
    axes[i].set_title(titles_cn[i-1], fontsize=14, pad=10, fontname="SimHei")
    # 中文x轴，英文y轴
    axes[i].set_xlabel("是否购买", fontsize=12, fontname="SimHei")
    axes[i].set_ylabel(col, fontsize=12, fontname="Times New Roman")

    # x轴刻度
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["未购买", "已购买"])
    for label in axes[i].get_xticklabels():
        label.set_fontname("SimHei")
        label.set_fontsize(11)

    # y轴刻度
    for label in axes[i].get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(10)

    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

# ========= 第6个子图留空 =========
axes[5].axis("off")

# 总标题
plt.suptitle("Revenue 分布与关键特征差异总览", fontsize=18, y=0.98, fontname="SimHei")

# 调整布局，避免文字显示不全
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



#2.1编码与基础预处理
# 复制数据，避免直接修改原始df
df_processed = df.copy()

# 1. 布尔变量转为0/1
bool_cols = ["Weekend", "Revenue"]
for col in bool_cols:
    df_processed[col] = df_processed[col].astype(int)

# 2. 类别变量做独热编码
categorical_cols = ["Month", "VisitorType"]
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# 3. 查看处理后的数据基本情况
print("处理后数据集形状：", df_processed.shape)
print("\n处理后前10个字段名：")
print(df_processed.columns[:10].tolist())

print("\n处理后后10个字段名：")
print(df_processed.columns[-10:].tolist())

display(df_processed.head())


#2.2构造两套任务数据并划分训练集和测试集
from sklearn.model_selection import train_test_split

# 1. 把所有 bool 列统一转成 int
bool_columns_after_encoding = df_processed.select_dtypes(include=["bool"]).columns.tolist()
for col in bool_columns_after_encoding:
    df_processed[col] = df_processed[col].astype(int)

print("当前所有字段的数据类型统计：")
print(df_processed.dtypes.value_counts())

# -----------------------------
# 2. 回归任务：预测 PageValues
# -----------------------------
X_reg = df_processed.drop(columns=["PageValues"])
y_reg = df_processed["PageValues"]

# -----------------------------
# 3. 分类任务：预测 Revenue
# 注意：分类任务中不能把 Revenue 自己作为特征
# 同时为了避免信息泄露，通常也不把 PageValues 放进去做演示版分类
# -----------------------------
X_clf = df_processed.drop(columns=["Revenue", "PageValues"])
y_clf = df_processed["Revenue"]

# -----------------------------
# 4. 划分训练集和测试集
# -----------------------------
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# -----------------------------
# 5. 输出结果查看
# -----------------------------
print("回归任务：")
print("X_reg shape =", X_reg.shape)
print("y_reg shape =", y_reg.shape)
print("X_train_reg =", X_train_reg.shape, " | X_test_reg =", X_test_reg.shape)
print("y_train_reg =", y_train_reg.shape, " | y_test_reg =", y_test_reg.shape)

print("\n分类任务：")
print("X_clf shape =", X_clf.shape)
print("y_clf shape =", y_clf.shape)
print("X_train_clf =", X_train_clf.shape, " | X_test_clf =", X_test_clf.shape)
print("y_train_clf =", y_train_clf.shape, " | y_test_clf =", y_test_clf.shape)

print("\n分类任务标签分布（训练集）：")
print(y_train_clf.value_counts(normalize=True))

print("\n分类任务标签分布（测试集）：")
print(y_test_clf.value_counts(normalize=True))



# 第2部分 3标准化
# 线性回归、逻辑回归使用标准化后的数据
# 决策树仍使用未标准化数据
# =============================
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. 回归任务标准化
# -----------------------------
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# -----------------------------
# 2. 分类任务标准化
# -----------------------------
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# -----------------------------
# 3. 输出结果查看
# -----------------------------
print("回归任务标准化后：")
print("X_train_reg_scaled shape =", X_train_reg_scaled.shape)
print("X_test_reg_scaled shape  =", X_test_reg_scaled.shape)

print("\n分类任务标准化后：")
print("X_train_clf_scaled shape =", X_train_clf_scaled.shape)
print("X_test_clf_scaled shape  =", X_test_clf_scaled.shape)

# 查看前3个特征在标准化后的均值和标准差（训练集）
print("\n回归任务训练集前3个特征标准化后的均值：")
print(np.round(X_train_reg_scaled[:, :3].mean(axis=0), 4))

print("回归任务训练集前3个特征标准化后的标准差：")
print(np.round(X_train_reg_scaled[:, :3].std(axis=0), 4))

print("\n分类任务训练集前3个特征标准化后的均值：")
print(np.round(X_train_clf_scaled[:, :3].mean(axis=0), 4))

print("分类任务训练集前3个特征标准化后的标准差：")
print(np.round(X_train_clf_scaled[:, :3].std(axis=0), 4))



# =============================
# 3.1 线性回归实操
# Cell 1：训练模型 + 输出系数
# 任务：用行为特征预测 PageValues
# 注意：特征中不包含 Revenue
# =============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. 重新构造回归任务数据（去掉 PageValues 和 Revenue）
X_reg = df_processed.drop(columns=["PageValues", "Revenue"])
y_reg = df_processed["PageValues"]

# 2. 划分训练集和测试集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 3. 标准化（线性回归适合使用标准化后的特征）
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 4. 训练线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg_scaled, y_train_reg)

# 5. 输出截距和系数
print("线性回归模型训练完成！")
print("回归任务数据形状：", X_reg.shape)
print("截距（intercept）:")
print(lin_reg.intercept_)

coef_df = pd.DataFrame({
    "Feature": X_reg.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n按系数从大到小排序后的前10个特征：")
display(coef_df.head(10))

print("\n按系数从小到大排序后的前10个特征：")
display(coef_df.tail(10).sort_values(by="Coefficient", ascending=True))



# =============================
# 3.2 线性回归实操
# 预测 + 指标
# =============================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. 在测试集上预测
y_pred_reg = lin_reg.predict(X_test_reg_scaled)

# 2. 计算回归指标
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("线性回归预测完成！")
print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# 3. 补充查看前10个真实值与预测值
result_reg_df = pd.DataFrame({
    "真实值": y_test_reg.values[:10],
    "预测值": y_pred_reg[:10]
})

print("\n前10个样本的真实值与预测值：")
display(result_reg_df)



# =============================
# 3.3 线性回归实操
# 可视化
# 包含：
# 1. 真实值 vs 预测值
# 2. 残差图
# 3. 系数条形图
# =============================

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 字体与风格设置
# -----------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei"]      # 全局中文默认黑体
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# -----------------------------
# 图1：真实值 vs 预测值
# -----------------------------
axes[0].scatter(
    y_test_reg, y_pred_reg,
    color="#A8D8EA",
    alpha=0.65,
    edgecolor="black"
)
axes[0].plot(
    [y_test_reg.min(), y_test_reg.max()],
    [y_test_reg.min(), y_test_reg.max()],
    color="red",
    linestyle="--",
    linewidth=2
)
axes[0].set_title("真实值 vs 预测值", fontsize=14, fontname="SimHei")
axes[0].set_xlabel("真实值 PageValues", fontsize=12, fontname="Times New Roman")
axes[0].set_ylabel("预测值 PageValues", fontsize=12, fontname="Times New Roman")

for label in axes[0].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[0].get_yticklabels():
    label.set_fontname("Times New Roman")

# -----------------------------
# 图2：残差图
# -----------------------------
residuals = y_test_reg - y_pred_reg
axes[1].scatter(
    y_pred_reg, residuals,
    color="#FCBAD3",
    alpha=0.65,
    edgecolor="black"
)
axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
axes[1].set_title("残差图", fontsize=14, fontname="SimHei")
axes[1].set_xlabel("预测值 PageValues", fontsize=12, fontname="Times New Roman")
axes[1].set_ylabel("残差", fontsize=12, fontname="SimHei")

for label in axes[1].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[1].get_yticklabels():
    label.set_fontname("Times New Roman")

# -----------------------------
# 图3：系数条形图（绝对值最大的前10个）
# -----------------------------
coef_plot_df = coef_df.copy()
coef_plot_df["AbsCoefficient"] = coef_plot_df["Coefficient"].abs()
coef_plot_df = coef_plot_df.sort_values(by="AbsCoefficient", ascending=False).head(10)
coef_plot_df = coef_plot_df.sort_values(by="Coefficient", ascending=True)

axes[2].barh(
    coef_plot_df["Feature"],
    coef_plot_df["Coefficient"],
    color="#AA96DA",
    edgecolor="black"
)
axes[2].set_title("系数绝对值前10特征", fontsize=14, fontname="SimHei")
axes[2].set_xlabel("Coefficient", fontsize=12, fontname="Times New Roman")
axes[2].set_ylabel("Feature", fontsize=12, fontname="Times New Roman")

for label in axes[2].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[2].get_yticklabels():
    label.set_fontname("Times New Roman")

# -----------------------------
# 总标题
# -----------------------------
fig.suptitle("线性回归结果可视化", fontsize=18, fontname="SimHei", y=1.02)

plt.tight_layout()
plt.show()



# =============================
# 4.1 逻辑回归实操
# 训练模型
# 任务：预测 Revenue（是否购买）
# =============================

from sklearn.linear_model import LogisticRegression

# 1. 训练逻辑回归模型
log_reg = LogisticRegression(
    max_iter=2000,
    random_state=42
)
log_reg.fit(X_train_clf_scaled, y_train_clf)

# 2. 输出训练完成信息
print("逻辑回归模型训练完成！")
print("类别标签：", log_reg.classes_)
print("截距（intercept）:")
print(log_reg.intercept_[0])

# 3. 输出系数
log_coef_df = pd.DataFrame({
    "Feature": X_clf.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n按系数从大到小排序后的前10个特征：")
display(log_coef_df.head(10))

print("\n按系数从小到大排序后的前10个特征：")
display(log_coef_df.tail(10).sort_values(by="Coefficient", ascending=True))



# =============================
# 4.2 逻辑回归实操
# 预测 + 分类指标
# =============================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 1. 预测类别
y_pred_clf = log_reg.predict(X_test_clf_scaled)

# 2. 预测概率（取“购买=1”的概率）
y_pred_proba = log_reg.predict_proba(X_test_clf_scaled)[:, 1]

# 3. 计算分类指标
acc = accuracy_score(y_test_clf, y_pred_clf)
prec = precision_score(y_test_clf, y_pred_clf)
rec = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
auc = roc_auc_score(y_test_clf, y_pred_proba)

print("逻辑回归预测完成！")
print(f"Accuracy = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall = {rec:.4f}")
print(f"F1-score = {f1:.4f}")
print(f"AUC = {auc:.4f}")

print("\n分类报告：")
print(classification_report(y_test_clf, y_pred_clf))

# 4. 查看前10个样本的真实值、预测值和预测概率
result_clf_df = pd.DataFrame({
    "真实值": y_test_clf.values[:10],
    "预测值": y_pred_clf[:10],
    "预测购买概率": y_pred_proba[:10]
})

print("\n前10个样本的真实值、预测值与预测概率：")
display(result_clf_df)



# =============================
# 4.3 逻辑回归实操
# 可视化（混淆矩阵、ROC、系数）
# =============================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# -----------------------------
# 字体与风格设置
# -----------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文黑体
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# -----------------------------
# 图1：混淆矩阵
# -----------------------------
cm = confusion_matrix(y_test_clf, y_pred_clf)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=axes[0],
    linewidths=0.5,
    linecolor="white"
)

axes[0].set_title("混淆矩阵", fontsize=14, fontname="SimHei")
axes[0].set_xlabel("预测类别", fontsize=12, fontname="SimHei")
axes[0].set_ylabel("真实类别", fontsize=12, fontname="SimHei")
axes[0].set_xticklabels(["未购买", "已购买"], fontname="SimHei")
axes[0].set_yticklabels(["未购买", "已购买"], fontname="SimHei", rotation=0)

# -----------------------------
# 图2：ROC 曲线
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test_clf, y_pred_proba)

axes[1].plot(fpr, tpr, color="#FCBAD3", linewidth=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)
axes[1].set_title("ROC 曲线", fontsize=14, fontname="SimHei")
axes[1].set_xlabel("False Positive Rate", fontsize=12, fontname="Times New Roman")
axes[1].set_ylabel("True Positive Rate", fontsize=12, fontname="Times New Roman")
axes[1].legend(prop={"family": "Times New Roman", "size": 10})

for label in axes[1].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[1].get_yticklabels():
    label.set_fontname("Times New Roman")

# -----------------------------
# 图3：系数前10特征（按绝对值）
# -----------------------------
log_coef_plot_df = log_coef_df.copy()
log_coef_plot_df["AbsCoefficient"] = log_coef_plot_df["Coefficient"].abs()
log_coef_plot_df = log_coef_plot_df.sort_values(by="AbsCoefficient", ascending=False).head(10)
log_coef_plot_df = log_coef_plot_df.sort_values(by="Coefficient", ascending=True)

axes[2].barh(
    log_coef_plot_df["Feature"],
    log_coef_plot_df["Coefficient"],
    color="#AA96DA",
    edgecolor="black"
)

axes[2].set_title("系数绝对值前10特征", fontsize=14, fontname="SimHei")
axes[2].set_xlabel("Coefficient", fontsize=12, fontname="Times New Roman")
axes[2].set_ylabel("Feature", fontsize=12, fontname="Times New Roman")

for label in axes[2].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[2].get_yticklabels():
    label.set_fontname("Times New Roman")

# -----------------------------
# 总标题
# -----------------------------
fig.suptitle("逻辑回归结果可视化", fontsize=18, fontname="SimHei", y=1.02)

plt.tight_layout()
plt.show()



# =============================
# 5.1 决策树：回归预测
# 任务：预测 PageValues
# =============================

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. 训练回归树
tree_reg = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)
tree_reg.fit(X_train_reg, y_train_reg)

# 2. 预测
y_pred_tree_reg = tree_reg.predict(X_test_reg)

# 3. 计算回归指标
mae_tree = mean_absolute_error(y_test_reg, y_pred_tree_reg)
mse_tree = mean_squared_error(y_test_reg, y_pred_tree_reg)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test_reg, y_pred_tree_reg)

print("决策树回归模型训练与预测完成！")
print(f"MAE  = {mae_tree:.4f}")
print(f"MSE  = {mse_tree:.4f}")
print(f"RMSE = {rmse_tree:.4f}")
print(f"R²   = {r2_tree:.4f}")

# 4. 查看前10个样本的真实值与预测值
result_tree_reg_df = pd.DataFrame({
    "真实值": y_test_reg.values[:10],
    "预测值": y_pred_tree_reg[:10]
})

print("\n前10个样本的真实值与预测值：")
display(result_tree_reg_df)




# =============================
# 5.2 决策树：分类预测
# 任务：预测 Revenue
# =============================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. 训练分类树
tree_clf = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
tree_clf.fit(X_train_clf, y_train_clf)

# 2. 预测
y_pred_tree_clf = tree_clf.predict(X_test_clf)
y_pred_tree_clf_proba = tree_clf.predict_proba(X_test_clf)[:, 1]

# 3. 计算分类指标
acc_tree = accuracy_score(y_test_clf, y_pred_tree_clf)
prec_tree = precision_score(y_test_clf, y_pred_tree_clf)
rec_tree = recall_score(y_test_clf, y_pred_tree_clf)
f1_tree = f1_score(y_test_clf, y_pred_tree_clf)

print("决策树分类模型训练与预测完成！")
print(f"Accuracy = {acc_tree:.4f}")
print(f"Precision = {prec_tree:.4f}")
print(f"Recall = {rec_tree:.4f}")
print(f"F1-score = {f1_tree:.4f}")

print("\n分类报告：")
print(classification_report(y_test_clf, y_pred_tree_clf))

# 4. 查看前10个样本的真实值与预测值
result_tree_clf_df = pd.DataFrame({
    "真实值": y_test_clf.values[:10],
    "预测值": y_pred_tree_clf[:10],
    "预测购买概率": y_pred_tree_clf_proba[:10]
})

print("\n前10个样本的真实值、预测值与预测概率：")
display(result_tree_clf_df)



# =============================
# 5.3 决策树：可视化
# 包含：
# 1. 分类树结构图
# 2. 回归树结构图
# 3. 分类树特征重要性
# 4. 回归树特征重要性
# 5. 回归树真实值 vs 预测值
# =============================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# -----------------------------
# 字体与风格设置
# -----------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文统一黑体
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# -----------------------------
# 1. 分类树结构图
# 注意：class_names 用英文，避免中文在树节点中乱码
# -----------------------------
plt.figure(figsize=(22, 11))
plot_tree(
    tree_clf,
    feature_names=X_clf.columns,
    class_names=["NoBuy", "Buy"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("决策树分类结构图", fontsize=16, fontname="SimHei", pad=15)
plt.tight_layout()
plt.show()

# -----------------------------
# 2. 回归树结构图
# -----------------------------
plt.figure(figsize=(22, 11))
plot_tree(
    tree_reg,
    feature_names=X_reg.columns,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("决策树回归结构图", fontsize=16, fontname="SimHei", pad=15)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. 分类树特征重要性
# -----------------------------
clf_importance_df = pd.DataFrame({
    "Feature": X_clf.columns,
    "Importance": tree_clf.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(9, 6))
ax1 = sns.barplot(
    data=clf_importance_df,
    x="Importance",
    y="Feature",
    color="#A8D8EA",
    edgecolor="black"
)
plt.title("分类树特征重要性 Top 10", fontsize=15, fontname="SimHei", pad=12)
plt.xlabel("Importance", fontsize=12, fontname="Times New Roman")
plt.ylabel("Feature", fontsize=12, fontname="Times New Roman")

for label in ax1.get_xticklabels():
    label.set_fontname("Times New Roman")
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")

plt.tight_layout()
plt.show()

# -----------------------------
# 4. 回归树特征重要性
# -----------------------------
reg_importance_df = pd.DataFrame({
    "Feature": X_reg.columns,
    "Importance": tree_reg.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(9, 6))
ax2 = sns.barplot(
    data=reg_importance_df,
    x="Importance",
    y="Feature",
    color="#FCBAD3",
    edgecolor="black"
)
plt.title("回归树特征重要性 Top 10", fontsize=15, fontname="SimHei", pad=12)
plt.xlabel("Importance", fontsize=12, fontname="Times New Roman")
plt.ylabel("Feature", fontsize=12, fontname="Times New Roman")

for label in ax2.get_xticklabels():
    label.set_fontname("Times New Roman")
for label in ax2.get_yticklabels():
    label.set_fontname("Times New Roman")

plt.tight_layout()
plt.show()

# -----------------------------
# 5. 回归树真实值 vs 预测值
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test_reg,
    y_pred_tree_reg,
    color="#AA96DA",
    alpha=0.65,
    edgecolor="black"
)
plt.plot(
    [y_test_reg.min(), y_test_reg.max()],
    [y_test_reg.min(), y_test_reg.max()],
    color="red",
    linestyle="--",
    linewidth=2
)
plt.title("回归树：真实值 vs 预测值", fontsize=15, fontname="SimHei", pad=12)
plt.xlabel("真实值 PageValues", fontsize=12, fontname="Times New Roman")
plt.ylabel("预测值 PageValues", fontsize=12, fontname="Times New Roman")

ax3 = plt.gca()
for label in ax3.get_xticklabels():
    label.set_fontname("Times New Roman")
for label in ax3.get_yticklabels():
    label.set_fontname("Times New Roman")

plt.tight_layout()
plt.show()



# =============================
# 6.1 分类模型对比
# 逻辑回归 vs 决策树分类
# =============================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

# -----------------------------
# 字体与风格设置
# -----------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# -----------------------------
# 1. 整理分类指标
# -----------------------------
clf_compare_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC"],
    "Logistic Regression": [acc, prec, rec, f1, auc],
    "Decision Tree": [acc_tree, prec_tree, rec_tree, f1_tree, roc_auc_score(y_test_clf, y_pred_tree_clf_proba)]
})

print("分类模型对比指标表：")
display(clf_compare_df)

# -----------------------------
# 2. 绘制分类指标柱状图
# -----------------------------
x = np.arange(len(clf_compare_df["Metric"]))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(
    x - width/2,
    clf_compare_df["Logistic Regression"],
    width,
    label="Logistic Regression",
    color="#A8D8EA",
    edgecolor="black"
)
axes[0].bar(
    x + width/2,
    clf_compare_df["Decision Tree"],
    width,
    label="Decision Tree",
    color="#FCBAD3",
    edgecolor="black"
)

axes[0].set_title("分类模型指标对比", fontsize=15, fontname="SimHei", pad=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(clf_compare_df["Metric"], fontname="Times New Roman", fontsize=11)
axes[0].set_ylabel("Score", fontsize=12, fontname="Times New Roman")
axes[0].legend(prop={"family": "Times New Roman", "size": 10})

for label in axes[0].get_yticklabels():
    label.set_fontname("Times New Roman")

# 在柱子顶部标数值
for i, value in enumerate(clf_compare_df["Logistic Regression"]):
    axes[0].text(i - width/2, value + 0.01, f"{value:.2f}", ha="center",
                 fontsize=9, fontname="Times New Roman")
for i, value in enumerate(clf_compare_df["Decision Tree"]):
    axes[0].text(i + width/2, value + 0.01, f"{value:.2f}", ha="center",
                 fontsize=9, fontname="Times New Roman")

# -----------------------------
# 3. ROC 曲线对比
# -----------------------------
fpr_log, tpr_log, _ = roc_curve(y_test_clf, y_pred_proba)
fpr_tree, tpr_tree, _ = roc_curve(y_test_clf, y_pred_tree_clf_proba)

axes[1].plot(fpr_log, tpr_log, color="#AA96DA", linewidth=2,
             label=f"Logistic Regression (AUC={auc:.3f})")
axes[1].plot(fpr_tree, tpr_tree, color="#FFD3B6", linewidth=2,
             label=f"Decision Tree (AUC={roc_auc_score(y_test_clf, y_pred_tree_clf_proba):.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)

axes[1].set_title("ROC 曲线对比", fontsize=15, fontname="SimHei", pad=12)
axes[1].set_xlabel("False Positive Rate", fontsize=12, fontname="Times New Roman")
axes[1].set_ylabel("True Positive Rate", fontsize=12, fontname="Times New Roman")
axes[1].legend(prop={"family": "Times New Roman", "size": 10})

for label in axes[1].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[1].get_yticklabels():
    label.set_fontname("Times New Roman")

plt.suptitle("逻辑回归与决策树分类模型对比", fontsize=18, fontname="SimHei", y=1.02)
plt.tight_layout()
plt.show()



# =============================
# 6.2 回归模型对比
# 线性回归 vs 决策树回归
# =============================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# -----------------------------
# 字体与风格设置
# -----------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文统一黑体
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# -----------------------------
# 1. 整理回归指标
# 注意：R² 改成更稳的 R2
# -----------------------------
reg_compare_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Linear Regression": [mae, rmse, r2],
    "Decision Tree Regressor": [mae_tree, rmse_tree, r2_tree]
})

print("回归模型对比指标表：")
display(reg_compare_df)

# -----------------------------
# 2. 绘图
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ========= 图1：回归指标柱状图 =========
x = np.arange(len(reg_compare_df["Metric"]))
width = 0.35

bars1 = axes[0].bar(
    x - width/2,
    reg_compare_df["Linear Regression"],
    width,
    label="Linear Regression",
    color="#A8D8EA",
    edgecolor="black"
)
bars2 = axes[0].bar(
    x + width/2,
    reg_compare_df["Decision Tree Regressor"],
    width,
    label="Decision Tree Regressor",
    color="#FCBAD3",
    edgecolor="black"
)

axes[0].set_title("回归模型指标对比", fontsize=15, fontname="SimHei", pad=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(reg_compare_df["Metric"], fontsize=11, fontname="Times New Roman")
axes[0].set_ylabel("Score", fontsize=12, fontname="Times New Roman")
axes[0].legend(prop={"family": "Times New Roman", "size": 10})

for label in axes[0].get_yticklabels():
    label.set_fontname("Times New Roman")

# 柱顶标数值
for bar in bars1:
    value = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        value + 0.03,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontname="Times New Roman"
    )

for bar in bars2:
    value = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        value + 0.03,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontname="Times New Roman"
    )

# ========= 图2：真实值 vs 预测值散点对比 =========
axes[1].scatter(
    y_test_reg, y_pred_reg,
    color="#A8D8EA",
    alpha=0.45,
    edgecolor="black",
    label="Linear Regression"
)
axes[1].scatter(
    y_test_reg, y_pred_tree_reg,
    color="#FCBAD3",
    alpha=0.45,
    edgecolor="black",
    label="Decision Tree Regressor"
)

min_val = min(y_test_reg.min(), y_pred_reg.min(), y_pred_tree_reg.min())
max_val = max(y_test_reg.max(), y_pred_reg.max(), y_pred_tree_reg.max())

axes[1].plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    linewidth=2
)

axes[1].set_title("真实值 vs 预测值对比", fontsize=15, fontname="SimHei", pad=12)
axes[1].set_xlabel("真实值 PageValues", fontsize=12, fontname="Times New Roman")
axes[1].set_ylabel("预测值 PageValues", fontsize=12, fontname="Times New Roman")
axes[1].legend(prop={"family": "Times New Roman", "size": 10})

for label in axes[1].get_xticklabels():
    label.set_fontname("Times New Roman")
for label in axes[1].get_yticklabels():
    label.set_fontname("Times New Roman")

# -----------------------------
# 总标题
# -----------------------------
fig.suptitle("线性回归与决策树回归模型对比", fontsize=18, fontname="SimHei", y=1.02)

plt.tight_layout()
plt.show()

