import os
import warnings
import json
import pickle
import time
import logging
from typing import Dict, List, Union, Tuple, Optional, Any, Callable, Set
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import mlflow
import boto3
import requests

# 机器学习相关
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold,
    cross_val_score, learning_curve, TimeSeriesSplit, LeaveOneGroupOut
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, log_loss,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve,
    auc, average_precision_score, explained_variance_score, max_error,
    median_absolute_error, mean_squared_log_error, mean_tweedie_deviance
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    QuantileTransformer, OneHotEncoder, LabelEncoder, label_binarize
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV, VarianceThreshold, chi2, f_classif,
    mutual_info_classif, f_regression, mutual_info_regression
)
from sklearn.utils.class_weight import compute_class_weight

# 超参数优化工具
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from bayes_opt import BayesianOptimization

# 特征工程和解释工具
import shap
import eli5
from eli5.sklearn import PermutationImportance
import pdpbox
from pdpbox import pdp
import lime
from lime import lime_tabular
from featuretools import dfs, EntitySet
import category_encoders as ce

# 可视化
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

# 分布式处理
import dask.dataframe as dd
import dask.array as da
import dask_ml.model_selection
from distributed import Client, LocalCluster

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SuperXGBoost')


class SuperXGBoost:
    """
    超级增强版XGBoost框架：集成了高级特征工程、超参数优化、模型评估和部署的端到端解决方案

    主要特性:
    1. 全自动特征工程和选择
    2. 多种先进超参数优化方法(Hyperopt, Optuna, Ray Tune, Bayesian Optimization)
    3. 自适应学习率和早停策略
    4. 高级模型解释技术(SHAP, ELI5, PDP, LIME)
    5. 分布式训练支持(Dask, Ray)
    6. 内存优化和大数据处理
    7. MLOps集成(MLflow, 模型部署API)
    8. 自动特征交互检测
    9. 异常检测和处理
    10. 自适应交叉验证
    11. 时间序列特征支持
    12. 多相似模型集成
    13. 动态权重优化
    14. GPU加速支持
    15. 丰富的可视化工具
    """

    def __init__(
            self,
            task_type: str = 'classification',
            objective: str = None,
            gpu_acceleration: bool = False,
            distributed: bool = False,
            log_level: str = 'INFO',
            random_state: int = 42,
            verbose: int = 1,
            experiment_tracking: bool = False,
            experiment_name: str = None,
            auto_feature_engineering: bool = False,
            auto_feature_selection: bool = False,
            memory_optimization: bool = False
    ):
        """
        初始化SuperXGBoost模型框架

        参数:
            task_type: 任务类型，'classification'、'regression'或'ranking'
            objective: XGBoost的目标函数，如果为None则根据task_type自动选择
            gpu_acceleration: 是否使用GPU加速
            distributed: 是否使用分布式训练
            log_level: 日志级别，'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            random_state: 随机数种子
            verbose: 详细程度，0=安静，1=进度信息，2=详细信息
            experiment_tracking: 是否使用MLflow追踪实验
            experiment_name: MLflow实验名称
            auto_feature_engineering: 是否自动执行特征工程
            auto_feature_selection: 是否自动执行特征选择
            memory_optimization: 是否启用内存优化模式处理大数据集
        """
        # 设置日志级别
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)

        # 基本配置
        self.task_type = task_type.lower()
        self.gpu_acceleration = gpu_acceleration
        self.distributed = distributed
        self.random_state = random_state
        self.verbose = verbose
        self.experiment_tracking = experiment_tracking
        self.auto_feature_engineering = auto_feature_engineering
        self.auto_feature_selection = auto_feature_selection
        self.memory_optimization = memory_optimization

        # 初始化组件
        self.model = None
        self.best_model = None
        self.feature_names = None
        self.feature_types = None
        self.feature_importances = None
        self.categorical_features = []
        self.numerical_features = []
        self.text_features = []
        self.datetime_features = []
        self.transformers = {}
        self.feature_interactions = {}
        self.shap_values = None
        self.evaluation_results = {}
        self.cv_results = None
        self.best_params = {}
        self.train_data = None
        self.orig_train_data = None
        self.data_schema = {}
        self.data_profile = {}
        self.feature_stats = {}
        self.model_artifacts = {}
        self.model_metadata = {}
        self.pipeline = None
        self.training_history = []
        self.prediction_stats = {}
        self.outlier_detector = None
        self.prediction_intervals = {}
        self.feature_tracking = {}
        self.interpretability_reports = {}

        # 设置目标函数
        if objective is None:
            if self.task_type == 'classification':
                self.objective = 'binary:logistic'
            elif self.task_type == 'regression':
                self.objective = 'reg:squarederror'
            elif self.task_type == 'ranking':
                self.objective = 'rank:pairwise'
            else:
                raise ValueError("task_type必须是'classification'、'regression'或'ranking'")
        else:
            self.objective = objective

        # GPU设置
        if self.gpu_acceleration:
            logger.info("启用GPU加速")
            if 'tree_method' not in self.params:
                self.params['tree_method'] = 'gpu_hist'
            if 'predictor' not in self.params:
                self.params['predictor'] = 'gpu_predictor'

        # 分布式设置
        if self.distributed:
            logger.info("启用分布式训练")
            if ray.is_initialized():
                logger.info("使用现有Ray集群")
            else:
                try:
                    ray.init()
                    logger.info("Ray初始化成功")
                except Exception as e:
                    logger.warning(f"Ray初始化失败: {e}. 将使用本地模式。")
                    self.distributed = False

        # MLflow实验跟踪设置
        if self.experiment_tracking:
            try:
                mlflow.set_experiment(
                    experiment_name or f"SuperXGBoost_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                logger.info(
                    f"MLflow实验跟踪已启用，实验名称: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name if mlflow.active_run() else experiment_name}")
            except Exception as e:
                logger.warning(f"MLflow设置失败: {e}. 实验跟踪将被禁用。")
                self.experiment_tracking = False

        # 默认参数
        self.default_params = {
            'objective': self.objective,
            'booster': 'gbtree',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 1.0,
            'colsample_bynode': 1.0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'base_score': 0.5,
            'random_state': self.random_state,
            'missing': np.nan,
            'n_jobs': -1,
            'importance_type': 'gain'
        }

        # 如果使用GPU，设置树方法
        if self.gpu_acceleration:
            self.default_params['tree_method'] = 'gpu_hist'
            self.default_params['predictor'] = 'gpu_predictor'

        # 复制默认参数
        self.params = self.default_params.copy()

        # 打印初始化信息
        if self.verbose > 0:
            logger.info(f"初始化SuperXGBoost({self.task_type})，目标函数: {self.objective}")
            if self.gpu_acceleration:
                logger.info("GPU加速已启用")
            if self.distributed:
                logger.info("分布式训练已启用")
            if self.auto_feature_engineering:
                logger.info("自动特征工程已启用")
            if self.auto_feature_selection:
                logger.info("自动特征选择已启用")
            if self.memory_optimization:
                logger.info("内存优化模式已启用，适合处理大数据集")

    def get_parameter_guide(self, param_type='all'):
        """
        获取参数调优指南

        参数:
            param_type: 参数类型，'all'、'basic'、'advanced'、'sampling'、'regularization'等

        返回:
            Dict: 参数指南字典
        """
        param_guide = {
            'basic': {
                'learning_rate': {
                    'description': '学习率，控制每棵树的贡献',
                    'recommended_range': [0.01, 0.3],
                    'impact': '降低可以减少过拟合，但需要更多树',
                    'tuning_priority': 'high',
                    'interaction': '与n_estimators密切相关'
                },
                'n_estimators': {
                    'description': '弱学习器(树)的数量',
                    'recommended_range': [50, 1000],
                    'impact': '增加可以提高性能但可能过拟合',
                    'tuning_priority': 'high',
                    'interaction': '与learning_rate密切相关'
                },
                'max_depth': {
                    'description': '树的最大深度',
                    'recommended_range': [3, 10],
                    'impact': '增加可以提高复杂度但可能过拟合',
                    'tuning_priority': 'high',
                    'interaction': '与min_child_weight和gamma相关'
                },
                'min_child_weight': {
                    'description': '子节点中所需的最小样本权重和',
                    'recommended_range': [1, 10],
                    'impact': '增加可以减少过拟合',
                    'tuning_priority': 'high',
                    'interaction': '与max_depth相关'
                }
            },
            'sampling': {
                'subsample': {
                    'description': '训练实例的子采样比例',
                    'recommended_range': [0.5, 1.0],
                    'impact': '降低可以减少过拟合并提高速度',
                    'tuning_priority': 'medium',
                    'interaction': '与colsample参数相关'
                },
                'colsample_bytree': {
                    'description': '构建每棵树时列的子采样比例',
                    'recommended_range': [0.5, 1.0],
                    'impact': '降低可以减少过拟合',
                    'tuning_priority': 'medium',
                    'interaction': '与subsample相关'
                },
                'colsample_bylevel': {
                    'description': '树的每一级上的列子采样比例',
                    'recommended_range': [0.5, 1.0],
                    'impact': '细粒度控制列采样',
                    'tuning_priority': 'low',
                    'interaction': '与colsample_bytree相关'
                },
                'colsample_bynode': {
                    'description': '每个节点上的列子采样比例',
                    'recommended_range': [0.5, 1.0],
                    'impact': '最细粒度控制列采样',
                    'tuning_priority': 'low',
                    'interaction': '与其他colsample参数相关'
                }
            },
            'regularization': {
                'gamma': {
                    'description': '在节点分裂时，损失函数减少值的最小阈值',
                    'recommended_range': [0, 1.0],
                    'impact': '增加可以减少过拟合',
                    'tuning_priority': 'medium',
                    'interaction': '与max_depth相关'
                },
                'reg_alpha': {
                    'description': 'L1正则化项',
                    'recommended_range': [0, 1.0],
                    'impact': '增加可以减少过拟合，对稀疏特征有利',
                    'tuning_priority': 'medium',
                    'interaction': '与reg_lambda相关'
                },
                'reg_lambda': {
                    'description': 'L2正则化项',
                    'recommended_range': [0, 1.0],
                    'impact': '增加可以减少过拟合',
                    'tuning_priority': 'medium',
                    'interaction': '与reg_alpha相关'
                }
            },
            'advanced': {
                'scale_pos_weight': {
                    'description': '正样本的权重，用于不平衡分类问题',
                    'recommended_range': [1, 'neg_count/pos_count'],
                    'impact': '调整可以处理类别不平衡',
                    'tuning_priority': 'high for imbalanced data',
                    'interaction': '影响整体模型偏差'
                },
                'base_score': {
                    'description': '所有实例的初始预测分数',
                    'recommended_range': [0.5, '目标均值'],
                    'impact': '调整可以加速收敛',
                    'tuning_priority': 'low',
                    'interaction': '与learning_rate相关'
                },
                'grow_policy': {
                    'description': '树生长策略',
                    'options': ['depthwise', 'lossguide'],
                    'impact': 'lossguide可能更快但可能降低准确性',
                    'tuning_priority': 'low',
                    'interaction': '与max_depth和tree_method相关'
                },
                'max_leaves': {
                    'description': '最大叶节点数，仅在grow_policy=lossguide时有效',
                    'recommended_range': [0, '2的max_depth次方'],
                    'impact': '控制模型复杂度',
                    'tuning_priority': 'low',
                    'interaction': '与grow_policy相关'
                },
                'max_bin': {
                    'description': '分箱的最大数量，用于tree_method=hist/gpu_hist',
                    'recommended_range': [256, 512],
                    'impact': '较小的值加速训练但可能降低准确性',
                    'tuning_priority': 'low',
                    'interaction': '与tree_method相关'
                },
                'monotone_constraints': {
                    'description': '特征单调性约束',
                    'format': '特征索引到约束映射，1表示增加，-1表示减少，0表示无约束',
                    'impact': '确保预测与某些特征单调相关',
                    'tuning_priority': 'domain-specific',
                    'interaction': '受特征值分布影响'
                }
            },
            'distributed': {
                'tree_method': {
                    'description': '树构建算法',
                    'options': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'],
                    'impact': 'hist和gpu_hist更快但可能牺牲一些准确性',
                    'recommendation': 'hist适用于大数据集，gpu_hist适用于GPU训练'
                },
                'updater': {
                    'description': '树更新器选择',
                    'options': ['grow_colmaker', 'grow_histmaker', 'grow_local_histmaker', 'grow_skmaker', 'sync',
                                'refresh', 'prune'],
                    'impact': '影响训练速度和内存使用',
                    'recommendation': '通常由tree_method自动设置'
                },
                'predictor': {
                    'description': '预测器类型',
                    'options': ['auto', 'cpu_predictor', 'gpu_predictor'],
                    'impact': 'gpu_predictor在GPU上更快',
                    'recommendation': '与硬件平台匹配'
                }
            }
        }

        # 根据请求的参数类型返回相应的指南
        if param_type.lower() == 'all':
            return param_guide
        elif param_type.lower() in param_guide:
            return {param_type.lower(): param_guide[param_type.lower()]}
        else:
            logger.warning(f"未知的参数类型: {param_type}. 返回所有参数指南。")
            return param_guide

    def print_parameter_guide(self, param_type='all'):
        """
        打印参数调优指南

        参数:
            param_type: 参数类型，'all'、'basic'、'advanced'、'sampling'、'regularization'等
        """
        param_guide = self.get_parameter_guide(param_type)

        print("=" * 100)
        print(f"SuperXGBoost 参数调优指南 - {param_type.upper()}")
        print("=" * 100)

        for group_name, group_params in param_guide.items():
            print(f"\n{group_name.upper()} 参数:")
            print("-" * 80)

            for param_name, param_info in group_params.items():
                print(f"• {param_name}:")
                for info_key, info_value in param_info.items():
                    if isinstance(info_value, list):
                        info_value = f"[{info_value[0]}, {info_value[1]}]"
                    print(f"  - {info_key}: {info_value}")
                print()

        print("\n调优顺序建议:")
        print("1. 首先调整基本参数: learning_rate, n_estimators, max_depth, min_child_weight")
        print("2. 然后调整正则化参数: gamma, reg_alpha, reg_lambda")
        print("3. 调整采样参数: subsample, colsample_bytree")
        print("4. 最后调整高级参数")

        print("\n高级调优技巧:")
        print("• 使用较高的learning_rate (0.1-0.3)和较多的树进行初始探索")
        print("• 针对大数据集，使用tree_method='hist'可以显著加速")
        print("• 对于严重不平衡的分类问题，调整scale_pos_weight")
        print("• 在找到最佳max_depth和min_child_weight后，降低learning_rate并增加n_estimators")

        print("=" * 100)

    def set_params(self, **params):
        """
        设置模型参数

        参数:
            **params: 键值对形式的参数

        返回:
            self: 当前对象实例
        """
        self.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)

        if self.verbose > 0:
            logger.info(f"更新参数: {params}")

        if self.experiment_tracking and mlflow.active_run():
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

        return self

    def reset_params(self):
        """
        重置参数为默认值

        返回:
            self: 当前对象实例
        """
        self.params = self.default_params.copy()

        if self.verbose > 0:
            logger.info("已重置参数为默认值")

        return self

    def profile_data(self, X, y=None, compute_correlations=True, detect_categorical=True):
        """
        分析数据集并生成配置文件

        参数:
            X: 特征数据
            y: 目标变量 (可选)
            compute_correlations: 是否计算相关性
            detect_categorical: 是否自动检测分类特征

        返回:
            Dict: 数据配置文件
        """
        start_time = time.time()

        if self.verbose > 0:
            logger.info("开始数据分析...")

        # 处理DataFrame和转换为DataFrame
        if isinstance(X, (pd.DataFrame, dd.DataFrame)):
            df = X.copy()
            if isinstance(df, dd.DataFrame) and not self.distributed:
                df = df.compute()
                logger.info("Dask DataFrame已转换为pandas DataFrame")
        else:
            if hasattr(X, 'dtype') or hasattr(X, 'shape'):
                # 处理numpy数组或类似数组
                col_names = [f'feature_{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=col_names)
                logger.info(f"数组已转换为DataFrame，使用生成的列名: {col_names[:5]}...")
            else:
                raise ValueError("X必须是pandas DataFrame、Dask DataFrame或numpy数组")

        profile = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': {},
            'data_types': {},
            'numerical_stats': {},
            'categorical_stats': {},
            'datetime_stats': {},
            'feature_types': {},
            'unique_counts': {},
            'warnings': []
        }

        # 处理目标变量
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                profile['target'] = {
                    'shape': y.shape,
                    'dtype': str(y.dtype)
                }

                # 分类目标变量统计
                if self.task_type == 'classification':
                    profile['target']['class_distribution'] = y.value_counts().to_dict()
                    profile['target']['class_balance'] = y.value_counts(normalize=True).to_dict()
                    profile['target']['num_classes'] = len(profile['target']['class_distribution'])

                # 回归目标变量统计
                elif self.task_type == 'regression':
                    profile['target']['min'] = float(y.min())
                    profile['target']['max'] = float(y.max())
                    profile['target']['mean'] = float(y.mean())
                    profile['target']['median'] = float(y.median())
                    profile['target']['std'] = float(y.std())
                    profile['target']['skew'] = float(y.skew())

                    # 检查异常值
                    q1 = y.quantile(0.25)
                    q3 = y.quantile(0.75)
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    lower_bound = q1 - 1.5 * iqr
                    outliers = ((y < lower_bound) | (y > upper_bound)).sum()
                    profile['target']['outliers_count'] = int(outliers)
                    profile['target']['outliers_percentage'] = float(outliers / len(y) * 100)

        # 基本特征分析
        for col in df.columns:
            # 数据类型
            dtype = str(df[col].dtype)
            profile['data_types'][col] = dtype

            # 缺失值
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            profile['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }

            # 唯一值计数
            try:
                unique_count = df[col].nunique()
                profile['unique_counts'][col] = unique_count
            except:
                profile['unique_counts'][col] = 'unknown'
                profile['warnings'].append(f"无法计算列 '{col}' 的唯一值数量")

            # 根据数据类型进行分类
            if pd.api.types.is_numeric_dtype(df[col]):
                # 数值型特征
                if detect_categorical and (
                        unique_count < 20 or
                        (unique_count < 100 and unique_count / len(df) < 0.05)
                ):
                    feature_type = 'categorical_as_numeric'
                    if col not in self.categorical_features:
                        self.categorical_features.append(col)
                else:
                    feature_type = 'numerical'
                    if col not in self.numerical_features:
                        self.numerical_features.append(col)

                # 数值统计
                try:
                    stats = df[col].describe().to_dict()
                    stats['skew'] = float(df[col].skew())
                    stats['kurtosis'] = float(df[col].kurtosis())

                    # 检查异常值
                    q1 = stats['25%']
                    q3 = stats['75%']
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    lower_bound = q1 - 1.5 * iqr
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    stats['outliers_count'] = int(outliers)
                    stats['outliers_percentage'] = float(outliers / len(df) * 100)

                    # 零值和负值
                    stats['zeros_count'] = int((df[col] == 0).sum())
                    stats['zeros_percentage'] = float(stats['zeros_count'] / len(df) * 100)
                    stats['negative_count'] = int((df[col] < 0).sum())
                    stats['negative_percentage'] = float(stats['negative_count'] / len(df) * 100)

                    profile['numerical_stats'][col] = stats
                except Exception as e:
                    profile['warnings'].append(f"计算 '{col}' 的数值统计时出错: {str(e)}")

            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # 分类或文本特征
                if unique_count is not None and unique_count <= 1000:
                    # 可能是分类特征
                    feature_type = 'categorical'
                    try:
                        value_counts = df[col].value_counts(dropna=False).head(20).to_dict()
                        profile['categorical_stats'][col] = {
                            'value_counts': value_counts,
                            'top_count': int(df[col].value_counts().max()),
                            'top_percentage': float(df[col].value_counts(normalize=True).max() * 100)
                        }
                        if col not in self.categorical_features:
                            self.categorical_features.append(col)
                    except Exception as e:
                        profile['warnings'].append(f"计算 '{col}' 的分类统计时出错: {str(e)}")
                else:
                    # 可能是文本特征
                    feature_type = 'text'
                    if col not in self.text_features:
                        self.text_features.append(col)

            elif pd.api.types.is_datetime64_dtype(df[col]):
                # 时间特征
                feature_type = 'datetime'
                try:
                    profile['datetime_stats'][col] = {
                        'min': str(df[col].min()),
                        'max': str(df[col].max()),
                        'range_days': (df[col].max() - df[col].min()).days
                    }
                    if col not in self.datetime_features:
                        self.datetime_features.append(col)
                except Exception as e:
                    profile['warnings'].append(f"计算 '{col}' 的日期时间统计时出错: {str(e)}")

            else:
                # 其他类型
                feature_type = 'unknown'
                profile['warnings'].append(f"未知的数据类型: {dtype} (列 '{col}')")

            profile['feature_types'][col] = feature_type

        # 相关性分析
        if compute_correlations and len(self.numerical_features) > 1:
            try:
                # 仅使用数值特征
                numeric_df = df[self.numerical_features].copy()

                # 计算相关性
                corr_matrix = numeric_df.corr(method='pearson')
                profile['correlations'] = {}

                # 查找高相关性对
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.8:
                            high_corr_pairs.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': float(corr_value)
                            })

                profile['correlations']['high_correlation_pairs'] = high_corr_pairs

                # 与目标变量的相关性（如果提供）
                if y is not None and self.task_type == 'regression':
                    if isinstance(y, pd.Series) and pd.api.types.is_numeric_dtype(y):
                        # 创建包含特征和目标的DataFrame
                        corr_df = numeric_df.copy()
                        corr_df['target'] = y

                        # 计算与目标的相关性
                        target_corr = corr_df.corr()['target'].drop('target').sort_values(
                            ascending=False
                        ).to_dict()

                        profile['correlations']['with_target'] = target_corr
            except Exception as e:
                profile['warnings'].append(f"计算相关性时出错: {str(e)}")

        # 计算执行时间
        profile['analysis_time'] = time.time() - start_time

        # 保存数据概况
        self.data_profile = profile

        # 保存特征统计
        self.feature_stats = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'text_features': self.text_features,
            'datetime_features': self.datetime_features
        }

        if self.verbose > 0:
            logger.info(f"数据分析完成，耗时 {profile['analysis_time']:.2f} 秒")
            logger.info(f"特征类型: 数值型({len(self.numerical_features)}), "
                        f"分类型({len(self.categorical_features)}), "
                        f"文本型({len(self.text_features)}), "
                        f"时间型({len(self.datetime_features)})")

        return profile

    def print_data_profile(self, profile=None, sections=None):
        """
        打印数据概况

        参数:
            profile: 数据概况字典，如果为None则使用存储的概况
            sections: 要打印的部分列表，如果为None则打印所有部分
        """
        if profile is None:
            profile = self.data_profile

        if profile is None:
            logger.warning("未找到数据概况，请先调用profile_data方法")
            return

        if sections is None:
            sections = [
                'overview', 'data_types', 'missing_values', 'numerical_stats',
                'categorical_stats', 'datetime_stats', 'target',
                'correlations', 'warnings'
            ]

        print("\n" + "=" * 80)
        print(f"数据概况报告")
        print("=" * 80)

        # 基本概述
        if 'overview' in sections:
            print("\n数据集概述:")
            print(f"• 行数: {profile['shape'][0]:,}")
            print(f"• 列数: {profile['shape'][1]:,}")
            print(f"• 内存使用: {profile['memory_usage'] / (1024 ** 2):.2f} MB")

            # 特征类型汇总
            feature_type_counts = {}
            for feature_type in profile['feature_types'].values():
                feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1

            print("\n特征类型分布:")
            for feature_type, count in feature_type_counts.items():
                print(f"• {feature_type}: {count} 列 ({count / profile['shape'][1] * 100:.1f}%)")

        # 数据类型
        if 'data_types' in sections:
            print("\n数据类型:")
            for col, dtype in list(profile['data_types'].items())[:20]:
                print(f"• {col}: {dtype}")

            if len(profile['data_types']) > 20:
                print(f"... 以及 {len(profile['data_types']) - 20} 个其他列")

        # 缺失值
        if 'missing_values' in sections and 'missing_values' in profile:
            print("\n缺失值:")
            missing_cols = {col: info for col, info in profile['missing_values'].items()
                            if info['count'] > 0}

            if missing_cols:
                # 按缺失值百分比排序
                sorted_missing = sorted(
                    missing_cols.items(),
                    key=lambda x: x[1]['percentage'],
                    reverse=True
                )

                for col, info in sorted_missing[:10]:
                    print(f"• {col}: {info['count']:,} 缺失值 ({info['percentage']:.2f}%)")

                if len(sorted_missing) > 10:
                    print(f"... 以及 {len(sorted_missing) - 10} 个其他列有缺失值")
            else:
                print("数据集中没有缺失值")

        # 数值特征统计
        if 'numerical_stats' in sections and 'numerical_stats' in profile:
            print("\n数值特征统计 (前5个):")
            for i, (col, stats) in enumerate(list(profile['numerical_stats'].items())[:5]):
                print(f"• {col}:")
                print(f"  - 范围: [{stats['min']:.4g}, {stats['max']:.4g}]")
                print(f"  - 均值/中位数: {stats['mean']:.4g} / {stats['50%']:.4g}")
                print(f"  - 标准差: {stats['std']:.4g}")
                print(f"  - 偏度/峰度: {stats['skew']:.4g} / {stats['kurtosis']:.4g}")
                if 'outliers_percentage' in stats:
                    print(f"  - 异常值: {stats['outliers_count']:,} ({stats['outliers_percentage']:.2f}%)")

        # 分类特征统计
        if 'categorical_stats' in sections and 'categorical_stats' in profile:
            print("\n分类特征统计 (前5个):")
            for i, (col, stats) in enumerate(list(profile['categorical_stats'].items())[:5]):
                print(f"• {col}:")
                print(f"  - 唯一值数量: {profile['unique_counts'].get(col, 'unknown')}")
                print(f"  - 最频繁值占比: {stats['top_percentage']:.2f}%")
                print(f"  - 前3个值: ", end="")
                top_values = list(stats['value_counts'].items())[:3]
                print(", ".join([f"{val} ({count:,})" for val, count in top_values]))

        # 日期时间特征
        if 'datetime_stats' in sections and 'datetime_stats' in profile:
            print("\n日期时间特征:")
            for col, stats in profile['datetime_stats'].items():
                print(f"• {col}:")
                print(f"  - 范围: {stats['min']} 到 {stats['max']}")
                print(f"  - 跨度: {stats['range_days']} 天")

        # 目标变量
        if 'target' in sections and 'target' in profile:
            print("\n目标变量:")
            target = profile['target']
            print(f"• 数据类型: {target['dtype']}")

            if self.task_type == 'classification':
                print(f"• 类别数量: {target['num_classes']}")
                print("• 类别分布:")
                sorted_classes = sorted(
                    target['class_balance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for cls, pct in sorted_classes:
                    print(f"  - {cls}: {target['class_distribution'][cls]:,} ({pct * 100:.2f}%)")

            elif self.task_type == 'regression':
                print(f"• 范围: [{target['min']:.4g}, {target['max']:.4g}]")
                print(f"• 均值/中位数: {target['mean']:.4g} / {target['median']:.4g}")
                print(f"• 标准差: {target['std']:.4g}")
                print(f"• 偏度: {target['skew']:.4g}")
                if 'outliers_count' in target:
                    print(f"• 异常值: {target['outliers_count']:,} ({target['outliers_percentage']:.2f}%)")

        # 相关性
        if 'correlations' in sections and 'correlations' in profile:
            if 'high_correlation_pairs' in profile['correlations']:
                high_corr = profile['correlations']['high_correlation_pairs']
                if high_corr:
                    print("\n高相关性特征对:")
                    for pair in high_corr[:5]:
                        print(f"• {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.4f}")

                    if len(high_corr) > 5:
                        print(f"... 以及 {len(high_corr) - 5} 其他高相关性对")

            if 'with_target' in profile['correlations']:
                print("\n与目标变量的相关性 (前10个):")
                target_corr = profile['correlations']['with_target']
                for feature, corr in list(sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True))[:10]:
                    print(f"• {feature}: {corr:.4f}")

        # 警告
        if 'warnings' in sections and 'warnings' in profile and profile['warnings']:
            print("\n警告:")
            for warning in profile['warnings']:
                print(f"• {warning}")

        print("\n" + "=" * 80)

    def handle_missing_values(self, X, strategy='auto', fill_values=None, columns=None,
                              categorical_strategy='most_frequent', numerical_strategy='mean',
                              max_missing_ratio=0.8, drop_threshold=0.8, knn_neighbors=5):
        """
        处理缺失值

        参数:
            X: 特征数据
            strategy: 处理策略，'auto', 'impute', 'drop_rows', 'drop_columns', 'knn'
            fill_values: 自定义填充值的字典 {列名: 填充值}
            columns: 要处理的列，如果为None则处理所有列
            categorical_strategy: 分类特征的填充策略，'most_frequent', 'constant'
            numerical_strategy: 数值特征的填充策略，'mean', 'median', 'constant', 'iterative'
            max_missing_ratio: 识别为高缺失率的阈值（针对auto策略）
            drop_threshold: 删除列的缺失率阈值
            knn_neighbors: KNN填充的邻居数

        返回:
            X_imputed: 处理后的数据
        """
        if self.verbose > 0:
            logger.info("开始处理缺失值...")

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, 'toarray'):  # 处理scipy稀疏矩阵
                X = pd.DataFrame(X.toarray())
            else:
                X = pd.DataFrame(X)

        X_copy = X.copy()

        # 如果未指定列，处理所有列
        if columns is None:
            columns = X_copy.columns.tolist()
        else:
            # 确保所有列都存在
            for col in columns:
                if col not in X_copy.columns:
                    raise ValueError(f"列 '{col}' 不在数据中")

        # 计算每列的缺失率
        missing_ratio = X_copy[columns].isna().mean()
        high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
        medium_missing_cols = missing_ratio[(missing_ratio > 0) & (missing_ratio <= max_missing_ratio)].index.tolist()

        if self.verbose > 0:
            if high_missing_cols:
                logger.info(f"高缺失率列({max_missing_ratio * 100}%以上): {len(high_missing_cols)}个")
            logger.info(f"中等缺失率列: {len(medium_missing_cols)}个")

        # 根据策略处理缺失值
        if strategy == 'auto':
            # 自动策略：删除高缺失率列，填充中等缺失率列
            if high_missing_cols and self.verbose > 0:
                logger.info(f"自动删除高缺失率列: {high_missing_cols[:5]}{'...' if len(high_missing_cols) > 5 else ''}")

            # 删除高缺失率列
            X_copy = X_copy.drop(columns=high_missing_cols)
            columns = [col for col in columns if col not in high_missing_cols]

            # 对剩余列进行填充
            return self.handle_missing_values(
                X_copy,
                strategy='impute',
                fill_values=fill_values,
                columns=columns,
                categorical_strategy=categorical_strategy,
                numerical_strategy=numerical_strategy
            )

        elif strategy == 'drop_columns':
            # 删除缺失率高于阈值的列
            cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
            if cols_to_drop:
                if self.verbose > 0:
                    logger.info(
                        f"删除缺失率高于{drop_threshold * 100}%的列: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
                X_copy = X_copy.drop(columns=cols_to_drop)
            return X_copy

        elif strategy == 'drop_rows':
            # 删除包含缺失值的行
            original_shape = X_copy.shape
            X_copy = X_copy.dropna(subset=columns)
            if self.verbose > 0:
                rows_dropped = original_shape[0] - X_copy.shape[0]
                logger.info(f"删除了{rows_dropped}行数据 ({rows_dropped / original_shape[0] * 100:.2f}%)")
            return X_copy

        elif strategy == 'knn':
            # KNN填充
            if self.verbose > 0:
                logger.info(f"使用KNN填充缺失值 (neighbors={knn_neighbors})...")

            # 分离数值列和非数值列
            numeric_cols = X_copy.select_dtypes(include=['number']).columns.tolist()
            non_numeric_cols = [col for col in X_copy.columns if col not in numeric_cols]

            # 只能对数值列使用KNN填充
            if numeric_cols:
                knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
                X_numeric = X_copy[numeric_cols]
                X_numeric_imputed = pd.DataFrame(
                    knn_imputer.fit_transform(X_numeric),
                    columns=numeric_cols,
                    index=X_copy.index
                )

                # 合并回非数值列
                if non_numeric_cols:
                    # 对非数值列使用常规填充
                    X_non_numeric = self.handle_missing_values(
                        X_copy[non_numeric_cols],
                        strategy='impute',
                        categorical_strategy=categorical_strategy
                    )
                    return pd.concat([X_numeric_imputed, X_non_numeric], axis=1)
                else:
                    return X_numeric_imputed
            else:
                logger.warning("没有数值列可以用于KNN填充，将使用常规填充")
                return self.handle_missing_values(
                    X_copy,
                    strategy='impute',
                    categorical_strategy=categorical_strategy
                )

        elif strategy == 'impute':
            # 填充缺失值
            if self.verbose > 0:
                logger.info(f"填充缺失值...")

            # 处理自定义填充值
            if fill_values is not None:
                for col, value in fill_values.items():
                    if col in X_copy.columns:
                        X_copy[col] = X_copy[col].fillna(value)
                        if self.verbose > 1:
                            logger.debug(f"使用自定义值 {value} 填充列 '{col}'")

            # 分组处理剩余的列
            remaining_cols = [col for col in columns if col in X_copy.columns]
            remaining_cols = [col for col in remaining_cols if X_copy[col].isna().any()]

            if not remaining_cols:
                return X_copy

            # 确定每列的数据类型和填充策略
            numeric_cols = []
            categorical_cols = []

            for col in remaining_cols:
                if col in self.categorical_features or pd.api.types.is_categorical_dtype(X_copy[col]):
                    categorical_cols.append(col)
                elif pd.api.types.is_numeric_dtype(X_copy[col]):
                    numeric_cols.append(col)
                else:
                    # 默认作为分类变量处理
                    categorical_cols.append(col)

            # 处理数值列
            if numeric_cols:
                if numerical_strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif numerical_strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif numerical_strategy == 'constant':
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                elif numerical_strategy == 'iterative':
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    imputer = IterativeImputer(random_state=self.random_state)
                else:
                    raise ValueError(f"无效的数值填充策略: {numerical_strategy}")

                # 填充
                X_numeric = X_copy[numeric_cols]
                X_numeric_imputed = pd.DataFrame(
                    imputer.fit_transform(X_numeric),
                    columns=numeric_cols,
                    index=X_copy.index
                )

                # 更新原始DataFrame
                for col in numeric_cols:
                    X_copy[col] = X_numeric_imputed[col]

                if self.verbose > 1:
                    logger.debug(f"使用 {numerical_strategy} 策略填充 {len(numeric_cols)} 个数值列")

            # 处理分类列
            if categorical_cols:
                if categorical_strategy == 'most_frequent':
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                elif categorical_strategy == 'constant':
                    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
                else:
                    raise ValueError(f"无效的分类填充策略: {categorical_strategy}")

                # 对每列单独处理以避免类型问题
                for col in categorical_cols:
                    col_data = X_copy[[col]]
                    X_copy[col] = cat_imputer.fit_transform(col_data)

                if self.verbose > 1:
                    logger.debug(f"使用 {categorical_strategy} 策略填充 {len(categorical_cols)} 个分类列")

            if self.verbose > 0:
                logger.info(f"缺失值处理完成")

            return X_copy

        else:
            raise ValueError(f"无效的缺失值处理策略: {strategy}")

    def detect_and_handle_outliers(self, X, method='iqr', contamination=0.05,
                                   columns=None, treatment='clip', return_mask=False):
        """
        检测和处理异常值

        参数:
            X: 特征数据
            method: 检测方法，'iqr', 'z_score', 'isolation_forest', 'lof', 'dbscan'
            contamination: 预期的异常值比例
            columns: 要检查的列，如果为None则检查所有数值列
            treatment: 处理方法，'clip', 'remove', 'impute', 'none'
            return_mask: 是否返回异常值掩码

        返回:
            X_cleaned: 处理后的数据
            outlier_mask: (可选) 异常值掩码
        """
        if self.verbose > 0:
            logger.info(f"开始使用 {method} 方法检测异常值...")

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_copy = X.copy()

        # 选择要检查的列
        if columns is None:
            # 仅检查数值列
            columns = X_copy.select_dtypes(include=['number']).columns.tolist()
        else:
            # 确保所有列都存在且为数值型
            for col in columns:
                if col not in X_copy.columns:
                    raise ValueError(f"列 '{col}' 不在数据中")
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    raise ValueError(f"列 '{col}' 不是数值型")

        # 初始化异常值掩码
        outlier_mask = pd.DataFrame(False, index=X_copy.index, columns=columns)

        if method == 'iqr':
            # 使用IQR方法检测异常值
            for col in columns:
                q1 = X_copy[col].quantile(0.25)
                q3 = X_copy[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask[col] = (X_copy[col] < lower_bound) | (X_copy[col] > upper_bound)

        elif method == 'z_score':
            # 使用Z-score方法检测异常值
            for col in columns:
                mean = X_copy[col].mean()
                std = X_copy[col].std()
                z_scores = (X_copy[col] - mean) / std
                outlier_mask[col] = abs(z_scores) > 3

        elif method == 'isolation_forest':
            # 使用Isolation Forest检测异常值
            from sklearn.ensemble import IsolationForest

            # 仅使用数值列进行异常值检测
            X_numeric = X_copy[columns]

            # 处理缺失值，因为Isolation Forest不能处理缺失值
            if X_numeric.isna().any().any():
                X_numeric = X_numeric.fillna(X_numeric.mean())

            model = IsolationForest(
                contamination=contamination,
                random_state=self.random_state
            )
            outliers = model.fit_predict(X_numeric)
            is_outlier = outliers == -1  # -1表示异常值

            # 为所有列设置相同的异常值标记
            outlier_mask = pd.DataFrame(
                np.repeat(is_outlier[:, np.newaxis], len(columns), axis=1),
                index=X_copy.index,
                columns=columns
            )

        elif method == 'lof':
            # 使用局部异常因子(LOF)检测异常值
            from sklearn.neighbors import LocalOutlierFactor

            X_numeric = X_copy[columns]
            if X_numeric.isna().any().any():
                X_numeric = X_numeric.fillna(X_numeric.mean())

            lof = LocalOutlierFactor(contamination=contamination)
            outliers = lof.fit_predict(X_numeric)
            is_outlier = outliers == -1

            outlier_mask = pd.DataFrame(
                np.repeat(is_outlier[:, np.newaxis], len(columns), axis=1),
                index=X_copy.index,
                columns=columns
            )

        elif method == 'dbscan':
            # 使用DBSCAN检测异常值
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            X_numeric = X_copy[columns]
            if X_numeric.isna().any().any():
                X_numeric = X_numeric.fillna(X_numeric.mean())

            # 标准化数据
            X_scaled = StandardScaler().fit_transform(X_numeric)

            # 使用DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X_scaled)
            is_outlier = clusters == -1  # -1表示噪声点

            outlier_mask = pd.DataFrame(
                np.repeat(is_outlier[:, np.newaxis], len(columns), axis=1),
                index=X_copy.index,
                columns=columns
            )

        else:
            raise ValueError(f"无效的异常值检测方法: {method}")

        # 统计每列的异常值数量
        outlier_counts = outlier_mask.sum()
        total_outliers = outlier_mask.any(axis=1).sum()

        if self.verbose > 0:
            logger.info(f"检测到 {total_outliers} 行包含异常值 ({total_outliers / len(X_copy) * 100:.2f}%)")
            if self.verbose > 1:
                for col in columns:
                    count = outlier_counts[col]
                    if count > 0:
                        logger.debug(f"列 '{col}': {count} 个异常值 ({count / len(X_copy) * 100:.2f}%)")

        # 应用异常值处理
        if treatment == 'none':
            # 不处理异常值
            pass

        elif treatment == 'remove':
            # 删除包含异常值的行
            rows_with_outliers = outlier_mask.any(axis=1)
            X_copy = X_copy[~rows_with_outliers]
            if self.verbose > 0:
                logger.info(f"删除了 {rows_with_outliers.sum()} 行异常值")

        elif treatment == 'clip':
            # 将异常值剪裁到上下限
            for col in columns:
                if outlier_counts[col] > 0:
                    q1 = X_copy[col].quantile(0.25)
                    q3 = X_copy[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # 应用剪裁
                    X_copy.loc[outlier_mask[col] & (X_copy[col] < lower_bound), col] = lower_bound
                    X_copy.loc[outlier_mask[col] & (X_copy[col] > upper_bound), col] = upper_bound

            if self.verbose > 0:
                logger.info(f"将异常值剪裁到IQR边界")

        elif treatment == 'impute':
            # 使用中位数填充异常值
            for col in columns:
                if outlier_counts[col] > 0:
                    median_value = X_copy[~outlier_mask[col]][col].median()
                    X_copy.loc[outlier_mask[col], col] = median_value

            if self.verbose > 0:
                logger.info(f"使用中位数填充异常值")

        else:
            raise ValueError(f"无效的异常值处理方法: {treatment}")

        # 保存异常值检测器
        self.outlier_detector = {
            'method': method,
            'contamination': contamination,
            'columns': columns,
            'outlier_counts': outlier_counts.to_dict(),
            'total_outliers': int(total_outliers)
        }

        if return_mask:
            return X_copy, outlier_mask
        else:
            return X_copy

    def engineer_features(self, X, y=None, operations=None, drop_original=False,
                          only_numeric=True, only_categorical=False, verbose=None):
        """
        自动特征工程

        参数:
            X: 特征数据
            y: 目标变量(可选)
            operations: 要应用的特征工程操作列表
            drop_original: 是否删除原始特征
            only_numeric: 是否仅处理数值特征
            only_categorical: 是否仅处理分类特征
            verbose: 详细程度

        返回:
            X_new: 增强的特征数据
        """
        if verbose is None:
            verbose = self.verbose

        if verbose > 0:
            logger.info("开始特征工程...")

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_copy = X.copy()

        # 如果未指定操作，使用默认操作集
        if operations is None:
            if only_categorical:
                operations = ['one_hot', 'target_encoding', 'label_encoding', 'count_encoding']
            elif only_numeric:
                operations = ['polynomial', 'log', 'sqrt', 'interactions', 'binning']
            else:
                operations = ['one_hot', 'polynomial', 'log', 'sqrt', 'interactions', 'binning']

        # 创建要处理的列列表
        if only_numeric:
            columns = X_copy.select_dtypes(include=['number']).columns.tolist()
        elif only_categorical:
            columns = []
            for col in X_copy.columns:
                if (col in self.categorical_features or
                        pd.api.types.is_categorical_dtype(X_copy[col]) or
                        pd.api.types.is_object_dtype(X_copy[col])):
                    columns.append(col)
        else:
            columns = X_copy.columns.tolist()

        if verbose > 0:
            logger.info(f"应用 {operations} 操作到 {len(columns)} 列")

        # 应用特征工程操作
        feature_mapping = {}  # 跟踪新特征与原始特征的映射

        # 1. 处理分类特征
        if 'one_hot' in operations and not only_numeric:
            cat_cols = []
            for col in columns:
                if (col in self.categorical_features or
                        pd.api.types.is_categorical_dtype(X_copy[col]) or
                        pd.api.types.is_object_dtype(X_copy[col])):

                    # 限制唯一值的数量以避免过多的one-hot列
                    if X_copy[col].nunique() < 20:
                        cat_cols.append(col)

            if cat_cols:
                if verbose > 0:
                    logger.info(f"对 {len(cat_cols)} 列应用独热编码")

                # 应用独热编码
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X_copy[cat_cols])

                # 创建新的DataFrame
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for col, cats in zip(cat_cols, encoder.categories_)
                             for cat in cats],
                    index=X_copy.index
                )

                # 添加到原始DataFrame
                X_copy = pd.concat([X_copy, encoded_df], axis=1)

                # 记录映射关系
                for col in cat_cols:
                    feature_mapping[col] = [c for c in encoded_df.columns if c.startswith(f"{col}_")]

                # 删除原始列
                if drop_original:
                    X_copy = X_copy.drop(columns=cat_cols)

        if 'label_encoding' in operations and not only_numeric:
            cat_cols = []
            for col in columns:
                if (col in self.categorical_features or
                        pd.api.types.is_categorical_dtype(X_copy[col]) or
                        pd.api.types.is_object_dtype(X_copy[col])):
                    cat_cols.append(col)

            if cat_cols:
                if verbose > 0:
                    logger.info(f"对 {len(cat_cols)} 列应用标签编码")

                for col in cat_cols:
                    # 应用标签编码
                    encoder = LabelEncoder()
                    X_copy[f"{col}_label"] = encoder.fit_transform(X_copy[col].astype(str))

                    # 记录映射关系
                    feature_mapping[col] = [f"{col}_label"]

                # 删除原始列
                if drop_original:
                    X_copy = X_copy.drop(columns=cat_cols)

        if 'target_encoding' in operations and not only_numeric and y is not None:
            cat_cols = []
            for col in columns:
                if (col in self.categorical_features or
                        pd.api.types.is_categorical_dtype(X_copy[col]) or
                        pd.api.types.is_object_dtype(X_copy[col])):
                    cat_cols.append(col)

            if cat_cols:
                if verbose > 0:
                    logger.info(f"对 {len(cat_cols)} 列应用目标编码")

                # 检查任务类型
                if self.task_type == 'regression':
                    # 回归任务使用平均值编码
                    for col in cat_cols:
                        # 计算每个类别的目标均值
                        means = X_copy.groupby(col)[y.name if isinstance(y, pd.Series) else 'target'].mean()

                        # 应用编码
                        X_copy[f"{col}_target_mean"] = X_copy[col].map(means)

                        # 记录映射关系
                        feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_target_mean"]
                else:
                    # 分类任务
                    target_col = y.name if isinstance(y, pd.Series) else 'target'
                    y_series = y if isinstance(y, pd.Series) else pd.Series(y, name=target_col)

                    # 合并特征和目标
                    data = pd.concat([X_copy, y_series], axis=1)

                    # 对每个分类特征应用目标编码
                    for col in cat_cols:
                        # 对于多分类问题，计算每个类别的概率
                        if len(np.unique(y)) > 2:
                            # 对每个目标类别创建编码
                            for target_class in np.unique(y):
                                # 计算属于该类别的概率
                                probs = data.groupby(col)[target_col].apply(
                                    lambda x: (x == target_class).mean()
                                )

                                # 应用编码
                                X_copy[f"{col}_target_prob_{target_class}"] = X_copy[col].map(probs)

                                # 记录映射关系
                                feature_mapping[col] = feature_mapping.get(col, []) + [
                                    f"{col}_target_prob_{target_class}"]
                        else:
                            # 二分类问题：计算正类的概率
                            probs = data.groupby(col)[target_col].mean()

                            # 应用编码
                            X_copy[f"{col}_target_prob"] = X_copy[col].map(probs)

                            # 记录映射关系
                            feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_target_prob"]

                # 删除原始列
                if drop_original:
                    X_copy = X_copy.drop(columns=cat_cols)

        if 'count_encoding' in operations and not only_numeric:
            cat_cols = []
            for col in columns:
                if (col in self.categorical_features or
                        pd.api.types.is_categorical_dtype(X_copy[col]) or
                        pd.api.types.is_object_dtype(X_copy[col])):
                    cat_cols.append(col)

            if cat_cols:
                if verbose > 0:
                    logger.info(f"对 {len(cat_cols)} 列应用计数编码")

                for col in cat_cols:
                    # 计算每个类别的频率
                    counts = X_copy[col].value_counts()

                    # 应用编码
                    X_copy[f"{col}_count"] = X_copy[col].map(counts)

                    # 记录映射关系
                    feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_count"]

                # 删除原始列
                if drop_original:
                    X_copy = X_copy.drop(columns=cat_cols)

        # 2. 处理数值特征
        numeric_cols = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(X_copy[col]):
                numeric_cols.append(col)

        if 'log' in operations and numeric_cols:
            if verbose > 0:
                logger.info(f"对 {len(numeric_cols)} 列应用对数变换")

            for col in numeric_cols:
                # 检查列是否适合对数变换（全部为正值）
                if (X_copy[col] > 0).all():
                    X_copy[f"{col}_log"] = np.log(X_copy[col])

                    # 记录映射关系
                    feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_log"]
                elif (X_copy[col] >= 0).all():
                    # 对于包含零的列，使用log1p
                    X_copy[f"{col}_log1p"] = np.log1p(X_copy[col])

                    # 记录映射关系
                    feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_log1p"]

        if 'sqrt' in operations and numeric_cols:
            if verbose > 0:
                logger.info(f"对 {len(numeric_cols)} 列应用平方根变换")

            for col in numeric_cols:
                # 检查列是否适合平方根变换（全部为非负值）
                if (X_copy[col] >= 0).all():
                    X_copy[f"{col}_sqrt"] = np.sqrt(X_copy[col])

                    # 记录映射关系
                    feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_sqrt"]

        if 'polynomial' in operations and numeric_cols:
            if verbose > 0:
                logger.info(f"对 {len(numeric_cols)} 列应用多项式变换")

            for col in numeric_cols:
                # 平方变换
                X_copy[f"{col}_squared"] = X_copy[col] ** 2

                # 立方变换
                X_copy[f"{col}_cubed"] = X_copy[col] ** 3

                # 记录映射关系
                feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_squared", f"{col}_cubed"]

        if 'binning' in operations and numeric_cols:
            if verbose > 0:
                logger.info(f"对 {len(numeric_cols)} 列应用分箱")

            for col in numeric_cols:
                # 使用分位数进行分箱
                X_copy[f"{col}_bin"] = pd.qcut(
                    X_copy[col],
                    q=10,
                    labels=False,
                    duplicates='drop'
                ).astype(int)

                # 记录映射关系
                feature_mapping[col] = feature_mapping.get(col, []) + [f"{col}_bin"]

        if 'interactions' in operations and len(numeric_cols) > 1:
            if verbose > 0:
                logger.info("创建特征交互项")

            # 限制交互项的数量以避免组合爆炸
            max_interactions = 20
            interaction_count = 0

            # 创建特征交互项（乘积）
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    if interaction_count >= max_interactions:
                        break

                    X_copy[f"{col1}_{col2}_interaction"] = X_copy[col1] * X_copy[col2]

                    # 记录映射关系
                    interaction_name = f"{col1}_{col2}_interaction"
                    feature_mapping[col1] = feature_mapping.get(col1, []) + [interaction_name]
                    feature_mapping[col2] = feature_mapping.get(col2, []) + [interaction_name]

                    interaction_count += 1

                if interaction_count >= max_interactions:
                    break

        # 删除原始特征（如果要求）
        if drop_original and not only_categorical and not only_numeric:
            # 确保只删除那些已经有了工程特征的列
            cols_to_drop = [col for col in columns if col in feature_mapping]
            if cols_to_drop:
                X_copy = X_copy.drop(columns=cols_to_drop)
                if verbose > 0:
                    logger.info(f"删除了 {len(cols_to_drop)} 个原始特征")

        # 保存特征映射
        self.feature_interactions.update(feature_mapping)

        if verbose > 0:
            orig_cols = len(columns)
            new_cols = len(X_copy.columns)
            logger.info(f"特征工程完成: {orig_cols} 个原始特征 -> {new_cols} 个特征")

        return X_copy

    def select_features(self, X, y, method='auto', n_features=None, threshold=None,
                        cv=5, scoring=None, direction='maximize'):
        """
        自动特征选择

        参数:
            X: 特征数据
            y: 目标变量
            method: 特征选择方法，'auto', 'variance', 'f_value', 'mutual_info', 'model_based', 'rfe', 'select_from_model'
            n_features: 要选择的特征数量
            threshold: 特征重要性阈值
            cv: 交叉验证折数
            scoring: 评分指标
            direction: 分数方向，'maximize'或'minimize'

        返回:
            X_selected: 选择的特征
            selected_features: 所选特征的名称列表
        """
        if self.verbose > 0:
            logger.info(f"开始使用 {method} 方法进行特征选择...")

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 确保y是numpy数组
        if isinstance(y, pd.Series):
            y = y.values

        # 初始化选择的特征
        selected_features = []

        # 设置默认参数
        if n_features is None and threshold is None:
            # 默认选择一半的特征
            n_features = X.shape[1] // 2

        # 根据任务类型选择默认评分指标
        if scoring is None:
            if self.task_type == 'classification':
                if len(np.unique(y)) > 2:
                    scoring = 'accuracy'
                else:
                    scoring = 'roc_auc'
            else:
                scoring = 'neg_mean_squared_error'

        # 特征选择方法
        if method == 'auto':
            # 自动选择最合适的方法
            if X.shape[1] > 1000:
                # 对于高维数据，使用方差过滤和模型选择的组合
                return self.select_features(
                    X, y, method='variance', threshold=0.01
                )
            elif self.task_type == 'classification':
                # 对于分类任务，使用互信息
                return self.select_features(
                    X, y, method='mutual_info', n_features=n_features
                )
            else:
                # 对于回归任务，使用模型选择
                return self.select_features(
                    X, y, method='model_based', n_features=n_features
                )

        elif method == 'variance':
            # 基于方差的特征选择
            if self.verbose > 0:
                logger.info("使用方差阈值进行特征选择")

            # 默认阈值
            if threshold is None:
                threshold = 0.01

            # 应用方差阈值
            selector = VarianceThreshold(threshold=threshold)
            X_selected = selector.fit_transform(X)

            # 获取所选特征的索引
            mask = selector.get_support()
            selected_features = X.columns[mask].tolist()

        elif method == 'f_value':
            # 基于F值的特征选择
            if self.verbose > 0:
                logger.info("使用F检验进行特征选择")

            # 根据任务类型选择F检验方法
            if self.task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression

            # 应用SelectKBest
            if n_features is not None:
                selector = SelectKBest(score_func=score_func, k=n_features)
            else:
                # 如果未指定特征数量，使用百分比阈值
                if threshold is None:
                    threshold = 0.1
                selector = SelectKBest(score_func=score_func, k='all')

            X_selected = selector.fit_transform(X, y)

            # 如果使用阈值，进行额外的筛选
            if n_features is None and threshold is not None:
                scores = selector.scores_
                if scores is not None:
                    # 标准化分数
                    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
                    # 应用阈值
                    mask = norm_scores > threshold
                    X_selected = X.loc[:, mask]
                    selected_features = X.columns[mask].tolist()
                else:
                    selected_features = X.columns[selector.get_support()].tolist()
            else:
                selected_features = X.columns[selector.get_support()].tolist()

        elif method == 'mutual_info':
            # 基于互信息的特征选择
            if self.verbose > 0:
                logger.info("使用互信息进行特征选择")

            # 根据任务类型选择互信息方法
            if self.task_type == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression

            # 应用SelectKBest
            if n_features is not None:
                selector = SelectKBest(score_func=score_func, k=n_features)
            else:
                # 如果未指定特征数量，使用百分比阈值
                if threshold is None:
                    threshold = 0.1
                selector = SelectKBest(score_func=score_func, k='all')

            X_selected = selector.fit_transform(X, y)

            # 如果使用阈值，进行额外的筛选
            if n_features is None and threshold is not None:
                scores = selector.scores_
                if scores is not None:
                    # 标准化分数
                    if scores.max() > scores.min():
                        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
                    else:
                        norm_scores = scores
                    # 应用阈值
                    mask = norm_scores > threshold
                    X_selected = X.loc[:, mask]
                    selected_features = X.columns[mask].tolist()
                else:
                    selected_features = X.columns[selector.get_support()].tolist()
            else:
                selected_features = X.columns[selector.get_support()].tolist()

        elif method == 'model_based':
            # 基于模型的特征选择
            if self.verbose > 0:
                logger.info("使用基于模型的特征选择")

            # 创建模型
            model = xgb.XGBClassifier(
                objective='binary:logistic' if self.task_type == 'classification' else 'reg:squarederror',
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                use_label_encoder=False,
                verbosity=0
            ) if self.task_type == 'classification' else xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                verbosity=0
            )

            # 训练模型并获取特征重要性
            model.fit(X, y)
            importances = model.feature_importances_

            # 根据重要性选择特征
            if n_features is not None:
                # 选择前n个特征
                indices = np.argsort(importances)[::-1][:n_features]
                selected_features = X.columns[indices].tolist()
                X_selected = X.iloc[:, indices]
            else:
                # 使用阈值
                if threshold is None:
                    threshold = 0.01
                mask = importances > threshold
                selected_features = X.columns[mask].tolist()
                X_selected = X.loc[:, mask]

        elif method == 'rfe':
            # 递归特征消除
            if self.verbose > 0:
                logger.info("使用递归特征消除进行特征选择")

            # 创建基础模型
            base_model = xgb.XGBClassifier(
                objective='binary:logistic' if self.task_type == 'classification' else 'reg:squarederror',
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                use_label_encoder=False,
                verbosity=0
            ) if self.task_type == 'classification' else xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                verbosity=0
            )

            # 确定特征数量
            if n_features is None:
                # 使用交叉验证自动确定最佳特征数量
                selector = RFECV(
                    estimator=base_model,
                    step=1,
                    cv=cv,
                    scoring=scoring,
                    min_features_to_select=5
                )
            else:
                # 使用指定的特征数量
                selector = RFE(
                    estimator=base_model,
                    n_features_to_select=n_features,
                    step=1
                )

            # 应用RFE
            X_selected = selector.fit_transform(X, y)

            # 获取所选特征
            selected_features = X.columns[selector.get_support()].tolist()

        elif method == 'select_from_model':
            # 基于模型的特征选择
            if self.verbose > 0:
                logger.info("使用SelectFromModel进行特征选择")

            # 创建模型
            model = xgb.XGBClassifier(
                objective='binary:logistic' if self.task_type == 'classification' else 'reg:squarederror',
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                use_label_encoder=False,
                verbosity=0
            ) if self.task_type == 'classification' else xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                verbosity=0
            )

            # 应用SelectFromModel
            if threshold is not None:
                selector = SelectFromModel(model, threshold=threshold)
            else:
                # 默认使用中位数作为阈值
                selector = SelectFromModel(model, max_features=n_features)

            X_selected = selector.fit_transform(X, y)

            # 获取所选特征
            selected_features = X.columns[selector.get_support()].tolist()

        else:
            raise ValueError(f"无效的特征选择方法: {method}")

        # 将结果转换为DataFrame
        if not isinstance(X_selected, pd.DataFrame):
            X_selected = pd.DataFrame(
                X_selected,
                columns=selected_features,
                index=X.index
            )

        if self.verbose > 0:
            logger.info(f"特征选择完成，从 {X.shape[1]} 个特征中选择了 {len(selected_features)} 个特征")
            if self.verbose > 1:
                logger.debug(f"选择的特征: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")

        return X_selected, selected_features

    def create_preprocessing_pipeline(self, categorical_cols=None, numerical_cols=None,
                                      text_cols=None, date_cols=None, impute_strategy='auto',
                                      scaling_strategy='auto', categorical_encoding='auto'):
        """
        创建预处理管道

        参数:
            categorical_cols: 分类特征列表
            numerical_cols: 数值特征列表
            text_cols: 文本特征列表
            date_cols: 日期特征列表
            impute_strategy: 缺失值填充策略
            scaling_strategy: 特征缩放策略
            categorical_encoding: 分类特征编码策略

        返回:
            sklearn Pipeline 对象
        """
        if self.verbose > 0:
            logger.info("创建预处理管道...")

        # 使用保存的特征信息
            # 使用保存的特征信息
            if categorical_cols is None and self.categorical_features:
                categorical_cols = self.categorical_features
            if numerical_cols is None and self.numerical_features:
                numerical_cols = self.numerical_features
            if text_cols is None and self.text_features:
                text_cols = self.text_features
            if date_cols is None and self.datetime_features:
                date_cols = self.datetime_features

            # 根据任务自动确定最佳策略
            if scaling_strategy == 'auto':
                scaling_strategy = 'standard'  # 默认使用标准化

            if categorical_encoding == 'auto':
                if self.task_type == 'classification':
                    categorical_encoding = 'onehot'
                else:
                    categorical_encoding = 'target'

            if impute_strategy == 'auto':
                # 根据数据集大小选择简单或KNN插补
                impute_strategy = 'simple'

            # 创建预处理转换器
            transformers = []

            # 1. 数值特征预处理
            if numerical_cols:
                num_pipeline = []

                # 1.1 缺失值处理
                if impute_strategy == 'simple':
                    num_pipeline.append(('imputer', SimpleImputer(strategy='median')))
                elif impute_strategy == 'knn':
                    num_pipeline.append(('imputer', KNNImputer(n_neighbors=5)))
                elif impute_strategy == 'iterative':
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    num_pipeline.append(('imputer', IterativeImputer(random_state=self.random_state)))

                # 1.2 特征缩放
                if scaling_strategy == 'standard':
                    num_pipeline.append(('scaler', StandardScaler()))
                elif scaling_strategy == 'minmax':
                    num_pipeline.append(('scaler', MinMaxScaler()))
                elif scaling_strategy == 'robust':
                    num_pipeline.append(('scaler', RobustScaler()))
                elif scaling_strategy == 'power':
                    num_pipeline.append(('scaler', PowerTransformer(method='yeo-johnson')))
                elif scaling_strategy == 'quantile':
                    num_pipeline.append(('scaler', QuantileTransformer(output_distribution='normal')))

                # 合并数值管道
                transformers.append(
                    ('num', Pipeline(steps=num_pipeline), numerical_cols)
                )

            # 2. 分类特征预处理
            if categorical_cols:
                cat_pipeline = []

                # 2.1 缺失值处理
                cat_pipeline.append(('imputer', SimpleImputer(strategy='most_frequent')))

                # 2.2 编码
                if categorical_encoding == 'onehot':
                    cat_pipeline.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
                elif categorical_encoding == 'ordinal':
                    cat_pipeline.append(('encoder', ce.OrdinalEncoder(handle_unknown='value')))
                elif categorical_encoding == 'target':
                    if self.task_type == 'classification':
                        cat_pipeline.append(('encoder', ce.TargetEncoder(handle_unknown='value')))
                    else:
                        cat_pipeline.append(('encoder', ce.TargetEncoder(handle_unknown='value')))
                elif categorical_encoding == 'binary':
                    cat_pipeline.append(('encoder', ce.BinaryEncoder(handle_unknown='value')))
                elif categorical_encoding == 'count':
                    cat_pipeline.append(('encoder', ce.CountEncoder(handle_unknown='value')))

                # 合并分类管道
                transformers.append(
                    ('cat', Pipeline(steps=cat_pipeline), categorical_cols)
                )

            # 3. 文本特征预处理
            if text_cols:
                from sklearn.feature_extraction.text import TfidfVectorizer

                # 简单的文本处理管道
                text_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                    ('tfidf', TfidfVectorizer(
                        max_features=1000,
                        stop_words='english',
                        min_df=2,
                        max_df=0.9
                    ))
                ])

                transformers.append(
                    ('text', text_pipeline, text_cols)
                )

            # 4. 日期特征预处理
            if date_cols:
                # 自定义日期特征提取器
                class DateFeatureExtractor(BaseEstimator, TransformerMixin):
                    def fit(self, X, y=None):
                        return self

                    def transform(self, X):
                        X_copy = X.copy()

                        # 确保所有列都是日期时间类型
                        for col in X_copy.columns:
                            if not pd.api.types.is_datetime64_dtype(X_copy[col]):
                                X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')

                        # 提取日期特征
                        result = pd.DataFrame(index=X_copy.index)

                        for col in X_copy.columns:
                            # 年、月、日
                            result[f"{col}_year"] = X_copy[col].dt.year
                            result[f"{col}_month"] = X_copy[col].dt.month
                            result[f"{col}_day"] = X_copy[col].dt.day

                            # 一周中的某天、一年中的某天
                            result[f"{col}_dayofweek"] = X_copy[col].dt.dayofweek
                            result[f"{col}_dayofyear"] = X_copy[col].dt.dayofyear

                            # 季度
                            result[f"{col}_quarter"] = X_copy[col].dt.quarter

                            # 是否月初/月末
                            result[f"{col}_is_month_start"] = X_copy[col].dt.is_month_start.astype(int)
                            result[f"{col}_is_month_end"] = X_copy[col].dt.is_month_end.astype(int)

                            # 是否季度初/季度末
                            result[f"{col}_is_quarter_start"] = X_copy[col].dt.is_quarter_start.astype(int)
                            result[f"{col}_is_quarter_end"] = X_copy[col].dt.is_quarter_end.astype(int)

                            # 是否年初/年末
                            result[f"{col}_is_year_start"] = X_copy[col].dt.is_year_start.astype(int)
                            result[f"{col}_is_year_end"] = X_copy[col].dt.is_year_end.astype(int)

                            # 小时、分钟(如果有时间成分)
                            if not (X_copy[col].dt.hour == 0).all():
                                result[f"{col}_hour"] = X_copy[col].dt.hour
                                result[f"{col}_minute"] = X_copy[col].dt.minute

                        return result

                date_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=np.datetime64('NaT'))),
                    ('extractor', DateFeatureExtractor())
                ])

                transformers.append(
                    ('date', date_pipeline, date_cols)
                )

            # 创建列转换器
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'  # 删除未指定的列
            )

            if self.verbose > 0:
                components = []
                if numerical_cols:
                    components.append(f"{len(numerical_cols)} 个数值特征")
                if categorical_cols:
                    components.append(f"{len(categorical_cols)} 个分类特征")
                if text_cols:
                    components.append(f"{len(text_cols)} 个文本特征")
                if date_cols:
                    components.append(f"{len(date_cols)} 个日期特征")

                logger.info(f"预处理管道已创建，处理 {', '.join(components)}")

            # 保存管道
            self.pipeline = preprocessor

            return preprocessor

        def prepare_data(self, X, y=None, test_size=0.2, val_size=None, stratify=True,
                         preprocessing=True, feature_engineering=False, feature_selection=False,
                         handle_outliers=False, handle_imbalance=False, return_test=True):
            """
            全面的数据准备

            参数:
                X: 特征数据
                y: 目标变量
                test_size: 测试集大小比例
                val_size: 验证集大小比例
                stratify: 是否使用分层抽样
                preprocessing: 是否应用预处理
                feature_engineering: 是否应用特征工程
                feature_selection: 是否应用特征选择
                handle_outliers: 是否处理异常值
                handle_imbalance: 是否处理类别不平衡
                return_test: 是否返回测试集

            返回:
                处理后的数据集
            """
            if self.verbose > 0:
                logger.info("开始全面数据准备流程...")

            # 转换为DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            # 转换目标变量
            if y is not None and not isinstance(y, pd.Series):
                y = pd.Series(y, name='target')

            # 保存原始数据
            X_orig = X.copy()

            # 1. 数据分析
            self.profile_data(X, y)

            # 2. 处理异常值
            if handle_outliers:
                X = self.detect_and_handle_outliers(X, method='iqr', treatment='clip')

            # 3. 特征工程
            if feature_engineering:
                X = self.engineer_features(X, y, drop_original=False)

            # 4. 划分训练集和测试集
            if y is not None and return_test:
                if self.verbose > 0:
                    logger.info(f"划分数据集: test_size={test_size}, val_size={val_size}")

                # 确定是否使用分层抽样
                stratify_param = None
                if stratify and self.task_type == 'classification':
                    stratify_param = y

                # 划分训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size,
                    stratify=stratify_param,
                    random_state=self.random_state
                )

                # 如果需要验证集，进一步划分
                if val_size is not None:
                    # 根据原始大小计算验证集比例
                    val_ratio = val_size / (1 - test_size)

                    # 确定是否使用分层抽样
                    stratify_val = None
                    if stratify and self.task_type == 'classification':
                        stratify_val = y_train

                    # 划分训练集和验证集
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train, y_train, test_size=val_ratio,
                        stratify=stratify_val,
                        random_state=self.random_state
                    )

                    if self.verbose > 0:
                        logger.info(
                            f"数据集划分: 训练集={len(X_train)}样本, 验证集={len(X_val)}样本, 测试集={len(X_test)}样本")
                else:
                    X_val, y_val = None, None

                    if self.verbose > 0:
                        logger.info(f"数据集划分: 训练集={len(X_train)}样本, 测试集={len(X_test)}样本")
            else:
                # 没有划分
                X_train, y_train = X, y
                X_test, y_test = None, None
                X_val, y_val = None, None

            # 5. 处理类别不平衡
            if handle_imbalance and y_train is not None and self.task_type == 'classification':
                if self.verbose > 0:
                    logger.info("处理类别不平衡...")

                # 计算类别权重
                classes = np.unique(y_train)
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y_train
                )

                # 创建权重字典
                self.class_weights = dict(zip(classes, class_weights))

                # 更新模型参数
                if len(classes) == 2:
                    # 二分类：使用scale_pos_weight
                    neg_count = np.sum(y_train == 0)
                    pos_count = np.sum(y_train == 1)

                    if pos_count > 0:
                        self.params['scale_pos_weight'] = neg_count / pos_count

                        if self.verbose > 0:
                            logger.info(f"设置scale_pos_weight={self.params['scale_pos_weight']:.4f}")
                else:
                    # 多分类：使用XGBoost的weight参数
                    self.sample_weights = np.ones(len(y_train))
                    for i, cls in enumerate(y_train):
                        self.sample_weights[i] = self.class_weights[cls]

                    if self.verbose > 0:
                        logger.info(f"创建样本权重向量，类别权重: {self.class_weights}")

            # 6. 特征选择
            if feature_selection and y_train is not None:
                if self.verbose > 0:
                    logger.info("执行特征选择...")

                X_train, selected_features = self.select_features(X_train, y_train, method='auto')

                # 更新测试集和验证集
                if X_test is not None:
                    X_test = X_test[selected_features]

                if X_val is not None:
                    X_val = X_val[selected_features]

                if self.verbose > 0:
                    logger.info(f"特征选择完成，保留 {len(selected_features)} 个特征")

            # 7. 预处理
            if preprocessing:
                if self.verbose > 0:
                    logger.info("应用预处理转换...")

                # 创建预处理管道
                if self.pipeline is None:
                    self.create_preprocessing_pipeline()

                # 应用预处理
                X_train_processed = pd.DataFrame(
                    self.pipeline.fit_transform(X_train, y_train),
                    index=X_train.index
                )

                # 更新测试集和验证集
                if X_test is not None:
                    X_test_processed = pd.DataFrame(
                        self.pipeline.transform(X_test),
                        index=X_test.index
                    )
                else:
                    X_test_processed = None

                if X_val is not None:
                    X_val_processed = pd.DataFrame(
                        self.pipeline.transform(X_val),
                        index=X_val.index
                    )
                else:
                    X_val_processed = None

                # 更新数据
                X_train = X_train_processed
                X_test = X_test_processed
                X_val = X_val_processed

                if self.verbose > 0:
                    logger.info(f"预处理完成，转换后特征数量: {X_train.shape[1]}")

            # 保存处理后的训练数据
            self.orig_train_data = X_orig
            self.train_data = X_train

            # 返回处理后的数据
            if X_val is not None and return_test:
                return X_train, X_val, X_test, y_train, y_val, y_test
            elif return_test:
                return X_train, X_test, y_train, y_test
            else:
                return X_train, y_train

        def fit(self, X, y, eval_set=None, sample_weight=None, early_stopping_rounds=None,
                feature_names=None, categorical_features=None, enable_categorical=False,
                callbacks=None):
            """
            训练XGBoost模型

            参数:
                X: 训练特征
                y: 目标变量
                eval_set: 评估集，用于早停
                sample_weight: 样本权重
                early_stopping_rounds: 早停轮数
                feature_names: 特征名称
                categorical_features: 分类特征索引
                enable_categorical: 是否启用分类特征原生支持
                callbacks: 回调函数列表

            返回:
                self: 当前对象实例
            """
            if self.verbose > 0:
                logger.info("开始训练XGBoost模型...")

            # 初始化MLflow跟踪
            if self.experiment_tracking and mlflow.active_run() is None:
                mlflow.start_run()

                # 记录参数
                for param_name, param_value in self.params.items():
                    mlflow.log_param(param_name, param_value)

            # 检查数据类型并转换
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()

                # 确定分类特征
                if categorical_features is None and enable_categorical:
                    categorical_features = []
                    for i, col in enumerate(X.columns):
                        if col in self.categorical_features or pd.api.types.is_categorical_dtype(X[col]):
                            categorical_features.append(i)

                X_values = X.values
            else:
                X_values = X

            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y

            # 更新参数
            params = self.params.copy()

            # 启用分类特征处理
            if enable_categorical and categorical_features:
                if self.verbose > 0:
                    logger.info(f"启用原生分类特征支持，{len(categorical_features)}个分类特征")

                params['enable_categorical'] = True

            # 创建DMatrix
            dtrain = xgb.DMatrix(
                data=X_values,
                label=y_values,
                weight=sample_weight,
                feature_names=feature_names,
                enable_categorical=enable_categorical
            )

            # 准备验证集
            evals = [(dtrain, 'train')]
            if eval_set is not None:
                for i, (X_eval, y_eval) in enumerate(eval_set):
                    if isinstance(X_eval, pd.DataFrame):
                        X_eval = X_eval.values

                    if isinstance(y_eval, pd.Series):
                        y_eval = y_eval.values

                    # 创建评估DMatrix
                    deval = xgb.DMatrix(
                        data=X_eval,
                        label=y_eval,
                        feature_names=feature_names,
                        enable_categorical=enable_categorical
                    )

                    evals.append((deval, f'eval_{i}'))

            # 准备回调
            if callbacks is None:
                callbacks = []

            # 添加预定义回调
            if self.experiment_tracking:
                # MLflow回调
                class MLflowCallback(xgb.callback.TrainingCallback):
                    def after_iteration(self, model, epoch, evals_log):
                        for eval_name, eval_metric in evals_log.items():
                            for metric_name, metric_values in eval_metric.items():
                                mlflow.log_metric(
                                    f"{eval_name}_{metric_name}",
                                    metric_values[-1],
                                    step=epoch
                                )
                        return False

                callbacks.append(MLflowCallback())

            # 设置适当的评估指标
            if 'eval_metric' not in params:
                if self.task_type == 'classification':
                    if 'multi' in self.objective:
                        params['eval_metric'] = ['mlogloss', 'merror']
                    else:
                        params['eval_metric'] = ['logloss', 'auc', 'error']
                else:
                    params['eval_metric'] = ['rmse', 'mae']

            # 训练模型
            self.training_history = {}

            start_time = time.time()

            if self.verbose > 0:
                logger.info(
                    f"开始训练: {params.get('n_estimators', 100)}棵树, learning_rate={params.get('learning_rate', 0.1)}")

            # 检查分布式训练
            if self.distributed:
                if self.verbose > 0:
                    logger.info("使用分布式训练")

                # 使用Dask进行分布式训练
                try:
                    import dask_xgboost as dxgb
                    import dask.array as da

                    # 转换为Dask数组
                    if isinstance(X_values, np.ndarray):
                        X_dask = da.from_array(X_values, chunks='auto')
                        y_dask = da.from_array(y_values, chunks='auto')
                    else:
                        X_dask = X_values
                        y_dask = y_values

                    # 分布式训练
                    self.model = dxgb.train(
                        client=None,  # 使用当前客户端
                        params=params,
                        dtrain=(X_dask, y_dask),
                        num_boost_round=params.get('n_estimators', 100),
                        evals=evals,
                        early_stopping_rounds=early_stopping_rounds,
                        callbacks=callbacks,
                        verbose_eval=self.verbose > 0
                    )
                except Exception as e:
                    logger.warning(f"分布式训练失败: {e}. 回退到标准训练。")
                    self.distributed = False

                    # 回退到标准训练
                    self.model = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=params.get('n_estimators', 100),
                        evals=evals,
                        early_stopping_rounds=early_stopping_rounds,
                        evals_result=self.training_history,
                        callbacks=callbacks,
                        verbose_eval=self.verbose > 0
                    )
            else:
                # 标准训练
                self.model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=params.get('n_estimators', 100),
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    evals_result=self.training_history,
                    callbacks=callbacks,
                    verbose_eval=self.verbose > 0
                )

            # 计算训练时间
            train_time = time.time() - start_time

            # 保存特征名称
            self.feature_names = feature_names

            # 获取特征重要性
            try:
                self.feature_importances = self.model.get_score(importance_type='gain')
            except:
                self.feature_importances = {}
                if self.verbose > 0:
                    logger.warning("无法获取特征重要性")

            # 记录训练后的指标
            if self.experiment_tracking:
                mlflow.log_metric("training_time", train_time)

                # 记录特征重要性
                if self.feature_importances:
                    # 记录图表
                    fig = self.plot_feature_importance(plot=False)
                    mlflow.log_figure(fig, "feature_importance.png")
                    plt.close(fig)

                    # 记录重要性值
                    for feature, importance in self.feature_importances.items():
                        mlflow.log_metric(f"importance_{feature}", importance)

            if self.verbose > 0:
                logger.info(f"模型训练完成，用时 {train_time:.2f} 秒")

                if early_stopping_rounds is not None and hasattr(self.model, 'best_iteration'):
                    logger.info(f"最佳迭代次数: {self.model.best_iteration}")

                if self.feature_importances:
                    top_features = sorted(
                        self.feature_importances.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    logger.info("前5个重要特征:")
                    for feature, importance in top_features:
                        logger.info(f"  - {feature}: {importance:.4f}")

            return self

        def predict(self, X, threshold=0.5, ntree_limit=None):
            """
            预测结果

            参数:
                X: 特征数据
                threshold: 分类阈值，仅用于二分类问题
                ntree_limit: 限制使用的树数量

            返回:
                预测结果
            """
            if self.model is None:
                raise ValueError("模型尚未训练，请先调用fit方法")

            # 检查并转换特征数据
            if isinstance(X, pd.DataFrame):
                # 检查是否需要应用预处理
                if self.pipeline is not None:
                    X_processed = self.pipeline.transform(X)
                else:
                    X_processed = X.values
            else:
                X_processed = X

            # 创建DMatrix
            dtest = xgb.DMatrix(X_processed, feature_names=self.feature_names)

            # 获取原始预测
            raw_preds = self.model.predict(dtest, ntree_limit=ntree_limit)

            # 处理预测结果
            if self.task_type == 'classification':
                if 'multi' in self.objective:
                    # 多分类：返回类别索引
                    return np.argmax(raw_preds, axis=1)
                else:
                    # 二分类：应用阈值
                    return (raw_preds > threshold).astype(int)
            else:
                # 回归：直接返回预测值
                return raw_preds

        def predict_proba(self, X, ntree_limit=None):
            """
            预测概率（仅用于分类问题）

            参数:
                X: 特征数据
                ntree_limit: 限制使用的树数量

            返回:
                预测概率
            """
            if self.task_type != 'classification':
                raise ValueError("predict_proba只适用于分类任务")

            if self.model is None:
                raise ValueError("模型尚未训练，请先调用fit方法")

            # 检查并转换特征数据
            if isinstance(X, pd.DataFrame):
                # 检查是否需要应用预处理
                if self.pipeline is not None:
                    X_processed = self.pipeline.transform(X)
                else:
                    X_processed = X.values
            else:
                X_processed = X

            # 创建DMatrix
            dtest = xgb.DMatrix(X_processed, feature_names=self.feature_names)

            # 获取原始预测
            raw_preds = self.model.predict(dtest, ntree_limit=ntree_limit)

            # 处理预测结果
            if 'multi' not in self.objective:
                # 二分类：返回两列 [1-p, p]
                return np.vstack([1 - raw_preds, raw_preds]).T
            else:
                # 多分类：直接返回
                return raw_preds

        def evaluate(self, X, y, threshold=0.5, metrics=None, detailed=False,
                     get_predictions=False, prediction_range=None):
            """
            评估模型性能

            参数:
                X: 特征数据
                y: 目标变量
                threshold: 分类阈值
                metrics: 要计算的指标列表
                detailed: 是否返回详细评估报告
                get_predictions: 是否返回预测值
                prediction_range: 预测值的限制范围[min, max]

            返回:
                评估指标字典
            """
            if self.model is None:
                raise ValueError("模型尚未训练，请先调用fit方法")

            if self.verbose > 0:
                logger.info("开始评估模型性能...")

            # 转换目标变量
            if isinstance(y, pd.Series):
                y = y.values

            # 根据任务类型设置默认指标
            if metrics is None:
                if self.task_type == 'classification':
                    if len(np.unique(y)) > 2:
                        # 多分类
                        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'log_loss']
                    else:
                        # 二分类
                        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
                else:
                    # 回归
                    metrics = ['rmse', 'mae', 'mse', 'r2', 'explained_variance', 'max_error']

            # 获取预测结果
            if self.task_type == 'classification':
                y_pred = self.predict(X, threshold=threshold)

                if 'multi' in self.objective or any(m in metrics for m in ['roc_auc', 'log_loss']):
                    y_prob = self.predict_proba(X)
                else:
                    # 二分类
                    y_prob = self.predict_proba(X)[:, 1]
            else:
                # 回归任务
                y_pred = self.predict(X)

                # 对预测应用范围限制
                if prediction_range is not None:
                    min_val, max_val = prediction_range
                    y_pred = np.clip(y_pred, min_val, max_val)

            # 计算评估指标
            eval_results = {}

            # 分类指标
            if self.task_type == 'classification':
                if 'accuracy' in metrics:
                    eval_results['accuracy'] = accuracy_score(y, y_pred)

                if 'precision' in metrics:
                    if len(np.unique(y)) > 2:
                        eval_results['precision_macro'] = precision_score(y, y_pred, average='macro')
                        eval_results['precision_micro'] = precision_score(y, y_pred, average='micro')
                        eval_results['precision_weighted'] = precision_score(y, y_pred, average='weighted')
                    else:
                        eval_results['precision'] = precision_score(y, y_pred)

                if 'recall' in metrics:
                    if len(np.unique(y)) > 2:
                        eval_results['recall_macro'] = recall_score(y, y_pred, average='macro')
                        eval_results['recall_micro'] = recall_score(y, y_pred, average='micro')
                        eval_results['recall_weighted'] = recall_score(y, y_pred, average='weighted')
                    else:
                        eval_results['recall'] = recall_score(y, y_pred)

                if 'f1' in metrics:
                    if len(np.unique(y)) > 2:
                        eval_results['f1_macro'] = f1_score(y, y_pred, average='macro')
                        eval_results['f1_micro'] = f1_score(y, y_pred, average='micro')
                        eval_results['f1_weighted'] = f1_score(y, y_pred, average='weighted')
                    else:
                        eval_results['f1'] = f1_score(y, y_pred)

                if 'roc_auc' in metrics:
                    if len(np.unique(y)) > 2:
                        try:
                            eval_results['roc_auc_ovr'] = roc_auc_score(y, y_prob, multi_class='ovr')
                            eval_results['roc_auc_ovo'] = roc_auc_score(y, y_prob, multi_class='ovo')
                        except:
                            if self.verbose > 0:
                                logger.warning("计算多分类ROC AUC时出错")
                    else:
                        try:
                            if isinstance(y_prob, np.ndarray) and len(y_prob.shape) > 1:
                                eval_results['roc_auc'] = roc_auc_score(y, y_prob[:, 1])
                            else:
                                eval_results['roc_auc'] = roc_auc_score(y, y_prob)
                        except:
                            if self.verbose > 0:
                                logger.warning("计算二分类ROC AUC时出错")

                if 'log_loss' in metrics:
                    try:
                        eval_results['log_loss'] = log_loss(y, y_prob)
                    except:
                        if self.verbose > 0:
                            logger.warning("计算log_loss时出错")

                if 'confusion_matrix' in metrics or detailed:
                    try:
                        eval_results['confusion_matrix'] = confusion_matrix(y, y_pred)
                    except:
                        if self.verbose > 0:
                            logger.warning("计算混淆矩阵时出错")

                if 'classification_report' in metrics or detailed:
                    try:
                        eval_results['classification_report'] = classification_report(y, y_pred, output_dict=True)
                    except:
                        if self.verbose > 0:
                            logger.warning("生成分类报告时出错")

                if 'average_precision' in metrics:
                    try:
                        if len(np.unique(y)) > 2:
                            # 对每个类别计算
                            y_bin = label_binarize(y, classes=np.unique(y))
                            eval_results['average_precision'] = {}

                            for i, cls in enumerate(np.unique(y)):
                                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                                eval_results['average_precision'][f'class_{cls}'] = ap
                        else:
                            if isinstance(y_prob, np.ndarray) and len(y_prob.shape) > 1:
                                ap = average_precision_score(y, y_prob[:, 1])
                            else:
                                ap = average_precision_score(y, y_prob)
                            eval_results['average_precision'] = ap
                    except Exception as e:
                        if self.verbose > 0:
                            logger.warning(f"计算平均精度时出错: {e}")

            # 回归指标
            else:
                if 'mse' in metrics:
                    eval_results['mse'] = mean_squared_error(y, y_pred)

                if 'rmse' in metrics:
                    eval_results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))

                if 'mae' in metrics:
                    eval_results['mae'] = mean_absolute_error(y, y_pred)

                if 'r2' in metrics:
                    eval_results['r2'] = r2_score(y, y_pred)

                if 'explained_variance' in metrics:
                    eval_results['explained_variance'] = explained_variance_score(y, y_pred)

                if 'max_error' in metrics:
                    eval_results['max_error'] = max_error(y, y_pred)

                if 'median_absolute_error' in metrics:
                    eval_results['median_absolute_error'] = median_absolute_error(y, y_pred)

                if 'mape' in metrics:
                    # 平均绝对百分比误差
                    try:
                        mask = y != 0
                        eval_results['mape'] = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
                    except Exception as e:
                        if self.verbose > 0:
                            logger.warning(f"计算MAPE时出错: {e}")

                if 'msle' in metrics:
                    # 均方对数误差
                    try:
                        if np.all(y >= 0) and np.all(y_pred >= 0):
                            eval_results['msle'] = mean_squared_log_error(y, y_pred)
                    except Exception as e:
                        if self.verbose > 0:
                            logger.warning(f"计算MSLE时出错: {e}")

            # 添加预测统计
            if detailed:
                pred_stats = {
                    'min': np.min(y_pred),
                    'max': np.max(y_pred),
                    'mean': np.mean(y_pred),
                    'std': np.std(y_pred),
                    'median': np.median(y_pred)
                }
                eval_results['prediction_stats'] = pred_stats

                # 错误分析
                if self.task_type == 'classification':
                    # 错误分类的样本分析
                    incorrect_mask = y_pred != y
                    incorrect_count = np.sum(incorrect_mask)

                    eval_results['error_analysis'] = {
                        'incorrect_count': int(incorrect_count),
                        'incorrect_percentage': float(incorrect_count / len(y) * 100)
                    }

                    # 分类别的分析
                    if len(np.unique(y)) <= 10:  # 限制类别数量
                        per_class = {}
                        for cls in np.unique(y):
                            cls_mask = y == cls
                            cls_correct = np.sum((y_pred == cls) & cls_mask)
                            cls_total = np.sum(cls_mask)

                            per_class[str(cls)] = {
                                'accuracy': float(cls_correct / cls_total if cls_total > 0 else 0),
                                'count': int(cls_total),
                                'correct': int(cls_correct)
                            }

                        eval_results['per_class_performance'] = per_class
                else:
                    # 回归误差分析
                    errors = y - y_pred
                    abs_errors = np.abs(errors)
                    percentiles = [10, 25, 50, 75, 90, 95, 99]

                    error_stats = {
                        'mean_error': float(np.mean(errors)),
                        'mean_abs_error': float(np.mean(abs_errors)),
                        'std_error': float(np.std(errors)),
                        'max_abs_error': float(np.max(abs_errors)),
                        'percentiles': {
                            f'p{p}': float(np.percentile(abs_errors, p)) for p in percentiles
                        }
                    }

                    eval_results['error_analysis'] = error_stats

            # 保存评估结果
            self.evaluation_results.update(eval_results)

            # 打印评估结果
            if self.verbose > 0:
                logger.info("模型评估结果:")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  - {metric}: {value:.4f}")
                    elif isinstance(value, dict):
                        continue  # 跳过复杂结构
                    elif isinstance(value, np.ndarray) and value.size <= 25:
                        logger.info(f"  - {metric}:\n{value}")

            # 记录到MLflow
            if self.experiment_tracking and mlflow.active_run():
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"eval_{metric}", value)

            # 返回结果
            if get_predictions:
                if self.task_type == 'classification' and ('roc_auc' in metrics or 'log_loss' in metrics):
                    return eval_results, y_pred, y_prob
                else:
                    return eval_results, y_pred
            else:
                return eval_results

        def cross_validate(self, X, y, cv=5, stratified=None, metrics=None,
                           early_stopping_rounds=None, verbose=None):
            """
            交叉验证

            参数:
                X: 特征数据
                y: 目标变量
                cv: 折数或交叉验证对象
                stratified: 是否使用分层抽样
                metrics: 评估指标
                early_stopping_rounds: 早停轮数
                verbose: 详细程度

            返回:
                cv_results: 交叉验证结果
            """
            if verbose is None:
                verbose = self.verbose

            if verbose > 0:
                logger.info(f"开始{cv}折交叉验证...")

            # 根据任务类型确定分层策略
            if stratified is None:
                stratified = (self.task_type == 'classification')

            # 转换为DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y

            # 确定评估指标
            if metrics is None:
                if self.task_type == 'classification':
                    if len(np.unique(y_values)) > 2:
                        metrics = ['accuracy', 'f1_macro']
                    else:
                        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                else:
                    metrics = ['rmse', 'mae', 'r2']

            # 创建交叉验证划分对象
            if isinstance(cv, int):
                if stratified and self.task_type == 'classification':
                    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                else:
                    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                # 使用传入的交叉验证对象
                cv_obj = cv

            # 存储每折的结果
            fold_results = []
            fold_models = []
            all_metrics = {}
            feature_importances = {}

            # 执行交叉验证
            for fold, (train_idx, val_idx) in enumerate(cv_obj.split(X, y_values if stratified else None)):
                if verbose > 0:
                    logger.info(f"训练折 {fold + 1}/{cv_obj.get_n_splits()}")

                # 划分训练集和验证集
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_values[train_idx], y_values[val_idx]

                # 训练模型
                model = self.__class__(
                    task_type=self.task_type,
                    objective=self.objective,
                    random_state=self.random_state,
                    verbose=0  # 关闭每折的详细输出
                )
                model.set_params(**self.params)

                # 训练模型
                eval_set = [(X_val, y_val)]
                model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)

                # 评估模型
                metrics_result = model.evaluate(X_val, y_val, metrics=metrics)

                # 收集结果
                fold_results.append(metrics_result)
                fold_models.append(model.model)

                # 收集特征重要性
                if model.feature_importances:
                    for feature, importance in model.feature_importances.items():
                        if feature not in feature_importances:
                            feature_importances[feature] = []
                        feature_importances[feature].append(importance)

                if verbose > 1:
                    logger.debug(f"折 {fold + 1} 评估结果:")
                    for metric, value in metrics_result.items():
                        if isinstance(value, (int, float)):
                            logger.debug(f"  - {metric}: {value:.4f}")

            # 汇总结果
            cv_results = {}

            # 计算平均指标
            for metric in fold_results[0].keys():
                if isinstance(fold_results[0][metric], (int, float)):
                    values = [fold[metric] for fold in fold_results]
                    cv_results[f"{metric}_mean"] = np.mean(values)
                    cv_results[f"{metric}_std"] = np.std(values)
                    cv_results[f"{metric}_values"] = values

                    all_metrics[metric] = values

            # 特征重要性汇总
            if feature_importances:
                mean_importances = {}
                for feature, values in feature_importances.items():
                    mean_importances[feature] = np.mean(values)

                cv_results['feature_importance'] = {
                    'mean': mean_importances,
                    'per_fold': feature_importances
                }

            # 保存交叉验证结果
            self.cv_results = cv_results

            # 打印结果摘要
            if verbose > 0:
                logger.info(f"{cv}折交叉验证结果:")
                for metric, values in all_metrics.items():
                    logger.info(f"  - {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

            # 记录到MLflow
            if self.experiment_tracking and mlflow.active_run():
                for metric, values in all_metrics.items():
                    mlflow.log_metric(f"cv_{metric}_mean", np.mean(values))
                    mlflow.log_metric(f"cv_{metric}_std", np.std(values))

            return cv_results

        def grid_search(self, X, y, param_grid, cv=5, scoring=None, n_jobs=-1,
                        refit=True, verbose=None):
            """
            网格搜索调参

            参数:
                X: 特征数据
                y: 目标变量
                param_grid: 参数网格
                cv: 交叉验证折数
                scoring: 评分指标
                n_jobs: 并行任务数
                refit: 是否使用最佳参数重新拟合模型
                verbose: 详细程度

            返回:
                best_params: 最佳参数
            """
            if verbose is None:
                verbose = self.verbose

            if verbose > 0:
                logger.info("开始网格搜索...")
                param_combinations = 1
                for param, values in param_grid.items():
                    param_combinations *= len(values)
                logger.info(f"参数组合总数: {param_combinations}")

            # 使用分层CV进行分类任务
            if self.task_type == 'classification':
                cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_obj = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # 设置默认评分指标
            if scoring is None:
                if self.task_type == 'classification':
                    if len(np.unique(y)) > 2:
                        scoring = 'accuracy'
                    else:
                        scoring = 'roc_auc'
                else:
                    scoring = 'neg_mean_squared_error'

            # 创建基础模型
            if self.task_type == 'classification':
                base_model = xgb.XGBClassifier(
                    objective=self.objective,
                    random_state=self.random_state,
                    verbosity=0,
                    use_label_encoder=False
                )
            else:
                base_model = xgb.XGBRegressor(
                    objective=self.objective,
                    random_state=self.random_state,
                    verbosity=0
                )

            # 创建网格搜索对象
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv_obj,
                n_jobs=n_jobs,
                refit=refit,
                verbose=max(0, verbose - 1)
            )

            # 执行网格搜索
            start_time = time.time()
            grid_search.fit(X, y)
            search_time = time.time() - start_time

            # 提取最佳参数
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            # 更新模型参数
            self.best_params = best_params
            self.params.update(best_params)

            if verbose > 0:
                logger.info(f"网格搜索完成，耗时 {search_time:.2f} 秒")
                logger.info(f"最佳参数: {best_params}")
                logger.info(f"最佳{scoring}分数: {best_score:.4f}")

            # 记录到MLflow
            if self.experiment_tracking and mlflow.active_run():
                mlflow.log_metric(f"grid_search_best_{scoring}", best_score)
                mlflow.log_metric("grid_search_time", search_time)

                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)

            # 如果refit为True，模型已经使用最佳参数重新训练
            if refit:
                self.model = grid_search.best_estimator_

                # 提取内部booster
                if hasattr(self.model, 'get_booster'):
                    self.model = self.model.get_booster()

                if verbose > 0:
                    logger.info("使用最佳参数重新拟合了模型")

            return best_params, best_score

        def random_search(self, X, y, param_distributions, n_iter=10, cv=5,
                          scoring=None, n_jobs=-1, refit=True, verbose=None):
            """
            随机搜索调参

            参数:
                X: 特征数据
                y: 目标变量
                param_distributions: 参数分布
                n_iter: 迭代次数
                cv: 交叉验证折数
                scoring: 评分指标
                n_jobs: 并行任务数
                refit: 是否使用最佳参数重新拟合模型
                verbose: 详细程度

            返回:
                best_params: 最佳参数
            """
            if verbose is None:
                verbose = self.verbose

            if verbose > 0:
                logger.info(f"开始随机搜索 (n_iter={n_iter})...")

            # 使用分层CV进行分类任务
            if self.task_type == 'classification':
                cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_obj = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # 设置默认评分指标
            if scoring is None:
                if self.task_type == 'classification':
                    if len(np.unique(y)) > 2:
                        scoring = 'accuracy'
                    else:
                        scoring = 'roc_auc'
                else:
                    scoring = 'neg_mean_squared_error'

            # 创建基础模型
            if self.task_type == 'classification':
                base_model = xgb.XGBClassifier(
                    objective=self.objective,
                    random_state=self.random_state,
                    verbosity=0,
                    use_label_encoder=False
                )
            else:
                base_model = xgb.XGBRegressor(
                    objective=self.objective,
                    random_state=self.random_state,
                    verbosity=0
                )

            # 创建随机搜索对象
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv_obj,
                random_state=self.random_state,
                n_jobs=n_jobs,
                refit=refit,
                verbose=max(0, verbose - 1)
            )

            # 执行随机搜索
            start_time = time.time()
            random_search.fit(X, y)
            search_time = time.time() - start_time

            # 提取最佳参数
            best_params = random_search.best_params_
            best_score = random_search.best_score_

            # 更新模型参数
            self.best_params = best_params
            self.params.update(best_params)

            if verbose > 0:
                logger.info(f"随机搜索完成，耗时 {search_time:.2f} 秒")
                logger.info(f"最佳参数: {best_params}")
                logger.info(f"最佳{scoring}分数: {best_score:.4f}")

            # 记录到MLflow
            if self.experiment_tracking and mlflow.active_run():
                mlflow.log_metric(f"random_search_best_{scoring}", best_score)
                mlflow.log_metric("random_search_time", search_time)

                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)

            # 如果refit为True，模型已经使用最佳参数重新训练
            if refit:
                self.model = random_search.best_estimator_

                # 提取内部booster
                if hasattr(self.model, 'get_booster'):
                    self.model = self.model.get_booster()

                if verbose > 0:
                    logger.info("使用最佳参数重新拟合了模型")

            return best_params, best_score

        def hyperopt_optimize(self, X, y, param_space, max_evals=50, cv=5,
                              scoring=None, early_stopping_rounds=None, verbose=None):
            """
            使用Hyperopt进行超参数优化

            参数:
                X: 特征数据
                y: 目标变量
                param_space: 参数空间
                max_evals: 最大评估次数
                cv: 交叉验证折数
                scoring: 评分指标
                early_stopping_rounds: 早停轮数
                verbose: 详细程度

            返回:
                best_params: 最佳参数
            """
            if verbose is None:
                verbose = self.verbose

            if verbose > 0:
                logger.info(f"开始Hyperopt优化 (max_evals={max_evals})...")

            # 设置默认评分指标
            if scoring is None:
                if self.task_type == 'classification':
                    if len(np.unique(y)) > 2:
                        scoring = 'accuracy'
                    else:
                        scoring = 'roc_auc'
                else:
                    scoring = 'neg_mean_squared_error'

            # 使用分层CV进行分类任务
            if self.task_type == 'classification':
                cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_obj = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # 定义目标函数
            def objective(params):
                # 修正参数类型
                for param, value in params.items():
                    if param in ['max_depth', 'min_child_weight', 'n_estimators']:
                        params[param] = int(value)

                # 当前参数集合
                current_params = self.params.copy()
                current_params.update(params)

                # 创建模型
                if self.task_type == 'classification':
                    model = xgb.XGBClassifier(
                        **current_params,
                        random_state=self.random_state,
                        verbosity=0,
                        use_label_encoder=False
                    )
                else:
                    model = xgb.XGBRegressor(
                        **current_params,
                        random_state=self.random_state,
                        verbosity=0
                    )

                try:
                    # 交叉验证
                    cv_scores = []
                    for train_idx, val_idx in cv_obj.split(X, y):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        # 训练模型
                        eval_set = [(X_val, y_val)]
                        model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )

                        # 评估模型
                        if self.task_type == 'classification':
                            if 'multi' in self.objective:
                                # 多分类
                                y_pred = model.predict(X_val)
                                if scoring == 'accuracy':
                                    score = accuracy_score(y_val, y_pred)
                                elif scoring == 'f1_macro':
                                    score = f1_score(y_val, y_pred, average='macro')
                                elif scoring == 'log_loss':
                                    y_prob = model.predict_proba(X_val)
                                    score = -log_loss(y_val, y_prob)  # 取负值，因为hyperopt最小化
                                else:
                                    # 默认使用准确率
                                    score = accuracy_score(y_val, y_pred)
                            else:
                                # 二分类
                                if scoring == 'roc_auc':
                                    y_prob = model.predict_proba(X_val)[:, 1]
                                    score = roc_auc_score(y_val, y_prob)
                                elif scoring == 'accuracy':
                                    y_pred = model.predict(X_val)
                                    score = accuracy_score(y_val, y_pred)
                                elif scoring == 'f1':
                                    y_pred = model.predict(X_val)
                                    score = f1_score(y_val, y_pred)
                                elif scoring == 'log_loss':
                                    y_prob = model.predict_proba(X_val)
                                    score = -log_loss(y_val, y_prob)  # 取负值
                                else:
                                    # 默认使用AUC
                                    y_prob = model.predict_proba(X_val)[:, 1]
                                    score = roc_auc_score(y_val, y_prob)
                        else:
                            # 回归
                            y_pred = model.predict(X_val)
                            if scoring == 'neg_mean_squared_error':
                                score = -mean_squared_error(y_val, y_pred)  # 取负值
                            elif scoring == 'neg_root_mean_squared_error':
                                score = -np.sqrt(mean_squared_error(y_val, y_pred))  # 取负值
                            elif scoring == 'neg_mean_absolute_error':
                                score = -mean_absolute_error(y_val, y_pred)  # 取负值
                            elif scoring == 'r2':
                                score = r2_score(y_val, y_pred)
                            else:
                                # 默认使用MSE
                                score = -mean_squared_error(y_val, y_pred)  # 取负值

                        cv_scores.append(score)

                    # 计算平均分数
                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)

                    # 对于hyperopt来说，要最小化目标函数，所以对于越大越好的指标，要取负值
                    if scoring in ['accuracy', 'roc_auc', 'f1', 'f1_macro', 'r2']:
                        hyperopt_score = -mean_score
                    else:
                        # 对于已经是负值的指标（如neg_mean_squared_error），保持原样
                        hyperopt_score = mean_score

                    return {
                        'loss': hyperopt_score,
                        'status': STATUS_OK,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'params': params
                    }
                except Exception as e:
                    if verbose > 1:
                        logger.debug(f"评估参数时出错: {e}")
                    return {
                        'loss': float('inf'),
                        'status': STATUS_OK,
                        'error': str(e),
                        'params': params
                    }

            # 执行优化
            start_time = time.time()
            trials = Trials()
            best = fmin(
                fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                verbose=verbose > 1,
                rstate=np.random.RandomState(self.random_state)
            )

            # 获取最佳参数
            best_params = space_eval(param_space, best)

            # 修正参数类型
            for param, value in best_params.items():
                if param in ['max_depth', 'min_child_weight', 'n_estimators']:
                    best_params[param] = int(value)

            # 获取最佳分数
            best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials if 'result' in t])
            best_score = trials.trials[best_trial_idx]['result'].get('mean_score', float('inf'))

            # 更新模型参数
            self.best_params = best_params
            self.params.update(best_params)

            search_time = time.time() - start_time

            if verbose > 0:
                logger.info(f"Hyperopt优化完成，耗时 {search_time:.2f} 秒")
                logger.info(f"最佳参数: {best_params}")
                logger.info(f"最佳{scoring}分数: {best_score:.4f}")

            # 记录到MLflow
            if self.experiment_tracking and mlflow.active_run():
                mlflow.log_metric(f"hyperopt_best_{scoring}", best_score)
                mlflow.log_metric("hyperopt_time", search_time)

                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)

            # 使用最佳参数重新训练模型
            if self.task_type == 'classification':
                model = xgb.XGBClassifier(
                    **self.params,
                    random_state=self.random_state,
                    verbosity=0 if verbose == 0 else 1,
                    use_label_encoder=False
                )
            else:
                model = xgb.XGBRegressor(
                    **self.params,
                    random_state=self.random_state,
                    verbosity=0 if verbose == 0 else 1
                )

            # 训练模型
            if early_stopping_rounds is not None:
                # 创建验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state,
                    stratify=y if self.task_type == 'classification' else None
                )

                eval_set = [(X_val, y_val)]
                model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
            else:
                model.fit(X, y)

            # 保存模型
            self.model = model.get_booster() if hasattr(model, 'get_booster') else model

            return best_params, best_score

        def optuna_optimize(self, X, y, param_space_fn, n_trials=50, cv=5,
                            scoring=None, early_stopping_rounds=None, verbose=None):
            """
            使用Optuna进行超参数优化

            参数:
                X: 特征数据
                y: 目标变量
                param_space_fn: 参数空间函数，接受trial对象并返回参数字典
                n_trials: 试验次数
                cv: 交叉验证折数
                scoring: 评分指标
                early_stopping_rounds: 早停轮数
                verbose: 详细程度

            返回:
                best_params: 最佳参数
            """
            if verbose is None:
                verbose = self.verbose

            if verbose > 0:
                logger.info(f"开始Optuna优化 (n_trials={n_trials})...")

            # 设置默认评分指标
            if scoring is None:
                if self.task_type == 'classification':
                    if len(np.unique(y)) > 2:
                        scoring = 'accuracy'
                    else:
                        scoring = 'roc_auc'
                else:
                    scoring = 'neg_mean_squared_error'

            # 使用分层CV进行分类任务
            if self.task_type == 'classification':
                cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_obj = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # 定义目标函数
            def objective(trial):
                # 获取参数
                params = param_space_fn(trial)

                # 当前参数集合
                current_params = self.params.copy()
                current_params.update(params)

                # 创建模型
                if self.task_type == 'classification':
                    model = xgb.XGBClassifier(
                        **current_params,
                        random_state=self.random_state,
                        verbosity=0,
                        use_label_encoder=False
                    )
                else:
                    model = xgb.XGBRegressor(
                        **current_params,
                        random_state=self.random_state,
                        verbosity=0
                    )

                # 交叉验证
                cv_scores = []
                for train_idx, val_idx in cv_obj.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # 训练模型
                    eval_set = [(X_val, y_val)]
                    model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )

                    # 评估模型
                    if self.task_type == 'classification':
                        if 'multi' in self.objective:
                            # 多分类
                            y_pred = model.predict(X_val)
                            if scoring == 'accuracy':
                                score = accuracy_score(y_val, y_pred)
                            elif scoring == 'f1_macro':
                                score = f1_score(y_val, y_pred, average='macro')
                            elif scoring == 'log_loss':
                                y_prob = model.predict_proba(X_val)
                                score = -log_loss(y_val, y_prob)  # Optuna最大化
                            else:
                                # 默认使用准确率
                                score = accuracy_score(y_val, y_pred)
                        else:
                            # 二分类
                            if scoring == 'roc_auc':
                                y_prob = model.predict_proba(X_val)[:, 1]
                                score = roc_auc_score(y_val, y_prob)
                            elif scoring == 'accuracy':
                                y_pred = model.predict(X_val)
                                score = accuracy_score(y_val, y_pred)
                            elif scoring == 'f1':
                                y_pred = model.predict(X_val)
                                score = f1_score(y_val, y_pred)
                            elif scoring == 'log_loss':
                                y_prob = model.predict_proba(X_val)
                                score = -log_loss(y_val, y_prob)  # Optuna最大化
                            else:
                                # 默认使用AUC
                                y_prob = model.predict_proba(X_val)[:, 1]
                                score = roc_auc_score(y_val, y_prob)
                    else:
                        # 回归
                        y_pred = model.predict(X_val)
                        if scoring == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_val, y_pred)  # Optuna最大化
                        elif scoring == 'neg_root_mean_squared_error':
                            score = -np.sqrt(mean_squared_error(y_val, y_pred))  # Optuna最大化
                        elif scoring == 'neg_mean_absolute_error':
                            score = -mean_absolute_error(y_val, y_pred)  # Optuna最大化
                        elif scoring == 'r2':
                            score = r2_score(y_val, y_pred)
                        else:
                            # 默认使用MSE
                            score = -mean_squared_error(y_val, y_pred)  # Optuna最大化

                    cv_scores.append(score)

                # 计算平均分数
                mean_score = np.mean(cv_scores)

                # Optuna默认最大化目标函数
                return mean_score

            # 创建学习器
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler
            )

            # 执行优化
            start_time = time.time()
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=verbose > 0
            )

            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value

            # 更新模型参数
            self.best_params = best_params
            self.params.update(best_params)

            search_time = time.time() - start_time

            if verbose > 0:
                logger.info(f"Optuna优化完成，耗时 {search_time:.2f} 秒")
                logger.info(f"最佳参数: {best_params}")
                logger.info(f"最佳{scoring}分数: {best_score:.4f}")

                # 打印参数重要性
                try:
                    param_importance = optuna.importance.get_param_importances(study)
                    if param_importance:
                        logger.info("参数重要性:")
                        for param, importance in param_importance.items():
                            logger.info(f"  - {param}: {importance:.4f}")
                except Exception as e:
                    if verbose > 1:
                        logger.debug(f"计算参数重要性时出错: {e}")

            # 记录到MLflow
            if self.experiment_tracking and mlflow.active_run():
                mlflow.log_metric(f"optuna_best_{scoring}", best_score)
                mlflow.log_metric("optuna_time", search_time)

                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)

                # 记录参数重要性
                try:
                    param_importance = optuna.importance.get_param_importances(study)
                    for param, importance in param_importance.items():
                        mlflow.log_metric(f"param_importance_{param}", importance)
                except:
                    pass

            # 使用最佳参数重新训练模型
            if self.task_type == 'classification':
                model = xgb.XGBClassifier(
                    **self.params,
                    random_state=self.random_state,
                    verbosity=0 if verbose == 0 else 1,
                    use_label_encoder=False
                )
            else:
                model = xgb.XGBRegressor(
                    **self.params,
                    random_state=self.random_state,
                    verbosity=0 if verbose == 0 else 1
                )

            # 训练模型
            if early_stopping_rounds is not None:
                # 创建验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state,
                    stratify=y if self.task_type == 'classification' else None
                )

                eval_set = [(X_val, y_val)]
                model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
            else:
                model.fit(X, y)

            # 保存模型
            self.model = model.get_booster() if hasattr(model, 'get_booster') else model

            # 保存学习曲线图
            if verbose > 0 and early_stopping_rounds is not None and hasattr(model, 'evals_result'):
                try:
                    # 创建学习曲线图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    results = model.evals_result()

                    for eval_name, eval_metrics in results.items():
                        for metric_name, metric_values in eval_metrics.items():
                            ax.plot(range(len(metric_values)), metric_values, label=f'{eval_name}-{metric_name}')

                    ax.set_xlabel('迭代次数')
                    ax.set_ylabel('指标值')
                    ax.set_title('学习曲线')
                    ax.legend()
                    ax.grid(True)

                    # 保存到MLflow
                    if self.experiment_tracking and mlflow.active_run():
                        mlflow.log_figure(fig, "learning_curve.png")

                    plt.close(fig)
                except Exception as e:
                    if verbose > 1:
                        logger.debug(f"绘制学习曲线时出错: {e}")

            return best_params, best_score

        def plot_feature_importance(self, top_n=20, importance_type='gain', figsize=(12, 8), plot=True):
            """
            绘制特征重要性

            参数:
                top_n: 显示前N个重要特征
                importance_type: 重要性类型，'gain'、'weight'或'cover'
                figsize: 图表大小
                plot: 是否立即绘制

            返回:
                fig: matplotlib图表对象
            """
            if self.model is None:
                raise ValueError("模型尚未训练，请先调用fit方法")

            # 获取特征重要性
            try:
                importances = self.model.get_score(importance_type=importance_type)
            except Exception as e:
                if isinstance(self.model, (xgb.XGBClassifier, xgb.XGBRegressor)):
                    # scikit-learn接口
                    if hasattr(self.model, 'feature_importances_'):
                        if self.feature_names is not None:
                            importances = {name: imp for name, imp in
                                           zip(self.feature_names, self.model.feature_importances_)}
                        else:
                            importances = {f'f{i}': imp for i, imp in enumerate(self.model.feature_importances_)}
                    else:
                        raise ValueError(f"无法获取特征重要性: {e}")
                else:
                    raise ValueError(f"无法获取特征重要性: {e}")

            # 转换为DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(importances.keys()),
                'Importance': list(importances.values())
            })

            # 排序
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # 限制特征数量
            if top_n is not None:
                importance_df = importance_df.head(top_n)

            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)

            # 绘制水平条形图
            bars = ax.barh(
                importance_df['Feature'][::-1],
                importance_df['Importance'][::-1],
                color='skyblue',
                edgecolor='navy',
                alpha=0.8
            )

            # 添加标题和标签
            ax.set_title(f'特征重要性 (前{len(importance_df)}个特征)', fontsize=14)
            ax.set_xlabel(f'重要性 ({importance_type})', fontsize=12)
            ax.set_ylabel('特征', fontsize=12)

            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                        va='center', fontsize=10)

            # 添加网格线
            ax.grid(axis='x', linestyle='--', alpha=0.6)

            # 调整布局
            plt.tight_layout()

            # 立即绘制或返回图表
            if plot:
                plt.show()

            return fig

        def plot_tree(self, tree_index=0, rankdir='LR', figsize=(20, 10)):
            """
            绘制决策树

            参数:
                tree_index: 要绘制的树索引
                rankdir: 图形方向，'LR'为水平，'TB'为垂直
                figsize: 图表大小

            返回:
                fig: matplotlib图表对象
            """
            if self.model is None:
                raise ValueError("模型尚未训练，请先调用fit方法")

            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)

            # 使用xgboost的plot_tree函数
            xgb.plot_tree(
                self.model,
                num_trees=tree_index,
                rankdir=rankdir,
                ax=ax
            )

            # 添加标题
            ax.set_title(f'决策树 #{tree_index}', fontsize=14)

            # 调整布局
            plt.tight_layout()

            return fig

        def plot_learning_curve(self, metric='auto', figsize=(12, 6)):
            """
            绘制学习曲线

            参数:
                metric: 要绘制的指标
                figsize: 图表大小

            返回:
                fig: matplotlib图表对象
            """
            if not self.training_history:
                raise ValueError("没有训练历史数据")

            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)

            # 如果未指定指标，根据任务类型选择默认指标
            if metric == 'auto':
                if self.task_type == 'classification':
                    if 'multi' in self.objective:
                        if 'eval_0-mlogloss' in self.training_history:
                            metric = 'mlogloss'
                        else:
                            metric = 'merror'
                    else:
                        if 'eval_0-logloss' in self.training_history:
                            metric = 'logloss'
                        else:
                            metric = 'error'
                else:
                    if 'eval_0-rmse' in self.training_history:
                        metric = 'rmse'
                    else:
                        metric = 'mae'

            # 收集所有指标
            metrics_to_plot = []
            for eval_key in self.training_history.keys():
                for metric_key in self.training_history[eval_key].keys():
                    if metric in metric_key:
                        metrics_to_plot.append((eval_key, metric_key))

            # 绘制学习曲线
            for eval_key, metric_key in metrics_to_plot:
                values = self.training_history[eval_key][metric_key]
                iterations = range(1, len(values) + 1)

                ax.plot(iterations, values, label=f'{eval_key}-{metric_key}')

            # 添加标题和标签
            ax.set_title(f'学习曲线 ({metric})', fontsize=14)
            ax.set_xlabel('迭代次数', fontsize=12)
            ax.set_ylabel('指标值', fontsize=12)

            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')

            # 调整布局
            plt.tight_layout()

            return fig

        def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=None,
                                  title='混淆矩阵', cmap=plt.cm.Blues, figsize=(10, 8)):
            """
            绘制混淆矩阵

            参数:
                y_true: 真实标签
                y_pred: 预测标签
                labels: 类别标签
                normalize: 是否归一化，'true'、'pred'、'all'或None
                title: 图表标题
                cmap: 颜色映射
                figsize: 图表大小

            返回:
                fig: matplotlib图表对象
            """
            if self.task_type != 'classification':
                raise ValueError("混淆矩阵仅适用于分类任务")

            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # 确定类别标签
            if labels is None:
                labels = np.unique(np.concatenate((y_true, y_pred)))

            # 归一化
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                title = title + ' (按真实标签归一化)'
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                title = title + ' (按预测标签归一化)'
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
                title = title + ' (全局归一化)'

            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)

            # 显示混淆矩阵
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.colorbar(im, ax=ax)

            # 添加标题和轴标签
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('预测标签', fontsize=12)
            ax.set_ylabel('真实标签', fontsize=12)

            # 设置刻度标签
            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # 添加数值标签
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")

            # 调整布局
            fig.tight_layout()

            return fig

        def plot_roc_curve(self, y_true, y_prob, class_index=None, figsize=(10, 8)):
            """
            绘制ROC曲线

            参数:
                y_true: 真实标签
                y_prob: 预测概率
                class_index: 多分类问题中要绘制的类别索引
                figsize: 图表大小

            返回:
                fig: matplotlib图表对象
            """
            if self.task_type != 'classification':
                raise ValueError("ROC曲线仅适用于分类任务")

            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)

            # 处理多分类
            if 'multi' in self.objective:
                if class_index is None:
                    # 为每个类别绘制ROC曲线
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    n_classes = y_true_bin.shape[1]

                    # 对每个类别计算ROC曲线和AUC
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)

                        # 绘制ROC曲线
                        ax.plot(fpr, tpr, lw=2,
                                label=f'类别 {i} (AUC = {roc_auc:.3f})')
                else:
                    # 绘制指定类别的ROC曲线
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

                    fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
                    roc_auc = auc(fpr, tpr)

                    # 绘制ROC曲线
                    ax.plot(fpr, tpr, lw=2,
                            label=f'类别 {class_index} (AUC = {roc_auc:.3f})')
            else:
                # 二分类
                if isinstance(y_prob, np.ndarray) and len(y_prob.shape) > 1:
                    y_prob = y_prob[:, 1]

                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                # 绘制ROC曲线
                ax.plot(fpr, tpr, lw=2,
                        label=f'ROC曲线 (AUC = {roc_auc:.3f})')

            # 绘制基准线
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

            # 设置轴范围
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # 添加标题和标签
            ax.set_title('接收者操作特征曲线', fontsize=14)
            ax.set_xlabel('假正例率', fontsize=12)
            ax.set_ylabel('真正例率', fontsize=12)

            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right')

            # 加载数据
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split

            data = load_breast_cancer()
            X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

            # 初始化并自动调优
            model = SuperXGBoost(
                task_type='classification',
                gpu_acceleration=True,
                experiment_tracking=True,
                auto_feature_engineering=True
            )

            # 自动数据准备和特征工程
            X_train, X_test, y_train, y_test = model.prepare_data(
                X, y,
                preprocessing=True,
                feature_engineering=True,
                handle_outliers=True
            )

            # 模型调优与训练
            best_params = model.auto_tune(X_train, y_train, method='optuna', n_trials=50)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20)

            # 模型评估和解释
            eval_results = model.evaluate(X_test, y_test)
            model.explain_model(X_test)

            # 保存与部署
            model.save_model('super_xgboost_model.json')
            model.save_report('model_report.md')
            model.deploy_model_api(port=8000)
