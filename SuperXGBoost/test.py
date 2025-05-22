#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperXGBoost 使用示例程序
演示SuperXGBoost库的主要功能，包括数据处理、特征工程、模型训练、超参数优化和评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 导入SuperXGBoost（假设已经保存为Xgboost.py文件）
from Xgboost import SuperXGBoost

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def demo_classification():
    """
    分类任务演示 - 修复版本
    """
    print("=" * 80)
    print("SuperXGBoost 分类任务演示")
    print("=" * 80)

    # 1. 加载数据集
    print("\n1. 加载乳腺癌数据集...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    print(f"数据集形状: {X.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"类别分布:\n{y.value_counts()}")

    # 2. 初始化SuperXGBoost模型
    print("\n2. 初始化SuperXGBoost模型...")
    model = SuperXGBoost(
        task_type='classification',
        objective='binary:logistic',
        gpu_acceleration=False,  # 如果有GPU可以设置为True
        experiment_tracking=False,  # 如果要使用MLflow可以设置为True
        auto_feature_engineering=True,
        auto_feature_selection=True,
        memory_optimization=False,
        verbose=1,
        random_state=42
    )

    # 3. 数据分析和概况
    print("\n3. 数据分析...")
    profile = model.profile_data(X, y, compute_correlations=True)
    model.print_data_profile(sections=['overview', 'missing_values', 'target'])

    # 4. 综合数据准备
    print("\n4. 数据准备（包括预处理、特征工程、异常值处理）...")
    X_train, X_test, y_train, y_test = model.prepare_data(
        X, y,
        test_size=0.2,
        preprocessing=True,
        feature_engineering=True,
        feature_selection=True,
        handle_outliers=True,
        handle_imbalance=True
    )

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")

    # 5. 模型训练
    print("\n5. 模型训练...")
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=20
    )

    # 6. 基础评估
    print("\n6. 模型评估...")
    eval_results = model.evaluate(
        X_test, y_test,
        metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        detailed=True,
        data_already_processed=True
    )

    # 7. 超参数优化（可选）
    print("\n7. 超参数优化...")

    # 定义参数空间
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # 网格搜索
    try:
        best_params, best_score = model.grid_search(
            X_train, y_train,
            param_grid=param_grid,
            cv=3,  # 减少CV折数以加速演示
            scoring='roc_auc'
        )

        print(f"最佳参数: {best_params}")
        print(f"最佳交叉验证得分: {best_score:.4f}")

        # 8. 使用最佳参数重新训练和评估
        print("\n8. 使用最佳参数重新评估...")
        final_results = model.evaluate(X_test, y_test, detailed=True)
    except Exception as e:
        print(f"超参数优化失败: {e}")
        final_results = eval_results

    # 9. 交叉验证
    print("\n9. 交叉验证...")
    try:
        cv_results = model.cross_validate(X_train, y_train, cv=5)
    except Exception as e:
        print(f"交叉验证失败: {e}")

    # 10. 可视化【修复版本】
    print("\n10. 生成可视化图表...")

    # 特征重要性图
    try:
        fig_importance = model.plot_feature_importance(top_n=15)  # 【修复】移除plot=False参数
        print("✅ 特征重要性图生成成功")
        plt.show()
    except Exception as e:
        print(f"❌ 绘制特征重要性图失败: {e}")

    # 学习曲线
    try:
        fig_learning = model.plot_learning_curve()  # 【修复】移除plot=False参数
        print("✅ 学习曲线生成成功")
        plt.show()
    except Exception as e:
        print(f"❌ 绘制学习曲线失败: {e}")

    # 混淆矩阵
    try:
        y_pred = model.predict(X_test)
        fig_cm = model.plot_confusion_matrix(y_test, y_pred)  # 【修复】移除plot=False参数
        print("✅ 混淆矩阵生成成功")
        plt.show()
    except Exception as e:
        print(f"❌ 绘制混淆矩阵失败: {e}")

    # ROC曲线
    try:
        y_prob = model.predict_proba(X_test)
        # 【修复】处理概率数组的形状
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        fig_roc = model.plot_roc_curve(y_test, y_prob_pos)  # 【修复】移除plot=False参数
        print("✅ ROC曲线生成成功")
        plt.show()
    except Exception as e:
        print(f"❌ 绘制ROC曲线失败: {e}")

    return model, eval_results


def demo_regression():
    """
    回归任务演示
    """
    print("=" * 80)
    print("SuperXGBoost 回归任务演示")
    print("=" * 80)

    # 1. 加载数据集（使用波士顿房价数据集的替代数据）
    print("\n1. 创建回归数据集...")
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )

    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')

    print(f"数据集形状: {X.shape}")
    print(f"目标变量统计:\n{y.describe()}")

    # 2. 初始化模型
    print("\n2. 初始化SuperXGBoost回归模型...")
    model = SuperXGBoost(
        task_type='regression',
        objective='reg:squarederror',
        verbose=1,
        random_state=42
    )

    # 3. 数据准备
    print("\n3. 数据准备...")
    X_train, X_test, y_train, y_test = model.prepare_data(
        X, y,
        test_size=0.2,
        preprocessing=True,
        feature_engineering=True
    )

    # 4. 模型训练
    print("\n4. 模型训练...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # 5. 模型评估
    print("\n5. 模型评估...")
    eval_results = model.evaluate(
        X_test, y_test,
        metrics=['rmse', 'mae', 'r2', 'explained_variance'],
        detailed=True
    )

    # 6. 可视化预测结果
    print("\n6. 可视化预测结果...")
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('回归预测结果')
    plt.grid(True, alpha=0.3)
    plt.show()

    return model, eval_results


def demo_multiclass():
    """
    多分类任务演示 - 修复版本
    """
    print("=" * 80)
    print("SuperXGBoost 多分类任务演示 - 修复版本")
    print("=" * 80)

    # 1. 加载数据集
    print("\n1. 加载红酒数据集...")
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    print(f"数据集形状: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    print(f"类别分布:\n{y.value_counts()}")

    # 2. 初始化模型
    print("\n2. 初始化SuperXGBoost多分类模型...")
    model = SuperXGBoost(
        task_type='classification',
        objective='multi:softprob',
        verbose=1,
        random_state=42
    )

    # 【重要修复】手动设置num_class参数（虽然fit方法会自动设置，但这里明确设置以确保）
    num_classes = len(np.unique(y))
    model.set_params(num_class=num_classes)
    print(f"设置类别数量: {num_classes}")

    # 3. 数据准备和训练
    print("\n3. 数据准备和训练...")
    X_train, X_test, y_train, y_test = model.prepare_data(X, y, test_size=0.2)

    try:
        model.fit(X_train, y_train)
        print("✅ 多分类模型训练成功！")
    except Exception as e:
        print(f"❌ 多分类模型训练失败: {e}")
        return None, None

    # 4. 评估
    print("\n4. 模型评估...")
    try:
        eval_results = model.evaluate(
            X_test, y_test,
            metrics=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        )
        print("✅ 多分类模型评估成功！")
    except Exception as e:
        print(f"❌ 多分类模型评估失败: {e}")
        eval_results = {}

    return model, eval_results


def demo_parameter_tuning():
    """
    高级参数调优演示
    """
    print("=" * 80)
    print("SuperXGBoost 高级参数调优演示")
    print("=" * 80)

    # 打印参数指南
    print("\n参数调优指南:")
    model = SuperXGBoost(task_type='classification')
    model.print_parameter_guide(param_type='basic')

    # 使用小数据集进行快速演示
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. 随机搜索演示
    print("\n1. 随机搜索演示...")
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'n_estimators': [100, 150, 200, 250, 300],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    try:
        best_params_random, best_score_random = model.random_search(
            X_train, y_train,
            param_distributions=param_distributions,
            n_iter=10,  # 减少迭代次数以加速演示
            cv=3
        )

        print(f"随机搜索最佳参数: {best_params_random}")
        print(f"随机搜索最佳得分: {best_score_random:.4f}")
    except Exception as e:
        print(f"随机搜索失败: {e}")

    return model


def main():
    """
    主函数：运行所有演示 - 修复版本
    """
    print("SuperXGBoost 库完整功能演示 - 修复版本")
    print("=" * 80)

    try:
        # 1. 分类任务演示
        print("\n开始分类任务演示...")
        clf_model, clf_results = demo_classification()

        # 2. 回归任务演示
        print("\n\n开始回归任务演示...")
        reg_model, reg_results = demo_regression()

        # 3. 多分类任务演示【修复版本】
        print("\n\n开始多分类任务演示...")
        multi_model, multi_results = demo_multiclass()

        # 4. 参数调优演示
        print("\n\n开始参数调优演示...")
        tuned_model = demo_parameter_tuning()

        print("\n" + "=" * 80)
        print("所有演示完成！")
        print("=" * 80)

        # 打印结果摘要
        print("\n📊 结果摘要:")
        if clf_results and 'accuracy' in clf_results:
            print(f"✅ 二分类准确率: {clf_results['accuracy']:.4f}")
        if reg_results and 'r2' in reg_results:
            print(f"✅ 回归R²分数: {reg_results['r2']:.4f}")
        if multi_results and 'accuracy' in multi_results:
            print(f"✅ 多分类准确率: {multi_results['accuracy']:.4f}")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()