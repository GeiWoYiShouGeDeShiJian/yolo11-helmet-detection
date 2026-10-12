#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 模型测试集评估脚本

该文件用于在测试集上评估训练好的 YOLOv11 模型性能，
计算并输出各项评估指标，包括 mAP50、mAP50-95、精确度和召回率，
并将评估结果保存为 CSV 文件。

评估配置：
- 模型：runs/detect/helmet_final_phase/weights/best.pt
- 数据集：dataset/data.yaml
- 评估分割：test
- 置信度阈值：0.001
- IoU 阈值：0.6
- 最大检测数：300
- 工作线程数：0（Windows 下避免多进程问题）
"""

# 导入必要的库


from ultralytics import YOLO
import pandas as pd
import numpy as np


def main():
    """
    在测试集上评估 YOLOv11 模型
    
    加载训练好的模型，在测试集上进行评估，
    计算各项性能指标，并将结果保存为 CSV 文件。
    """
    # 1. 加载最终模型
    print("加载训练好的最佳模型...")
    model_path = "runs/detect/helmet_final_phase/weights/best.pt"
    model = YOLO(model_path)

    # 2. 在测试集上进行最终评估
    print("正在测试集上进行最终评估...")

    metrics = model.val(
        data="dataset/data.yaml",  # 数据集配置文件
        split="test",  # 使用测试集进行评估
        save_json=True,  # 保存评估结果为 JSON 文件
        conf=0.001,  # 置信度阈值
        iou=0.6,  # IoU 阈值
        max_det=300,  # 最大检测数
        workers=0,  # Windows 下设为 0 避免多进程问题
        name="final_test_evaluation"  # 评估结果保存名称
    )

    # 3. 打印并整理关键指标
    print("\n" + "=" * 60)
    print("【安全帽检测模型 - 测试集最终性能报告】")
    print("=" * 60)

    # 提取关键指标
    # 注意：新版本 Ultralytics 中，metrics.box.p 和 metrics.box.r 是数组
    precision_value = float(metrics.box.p) if not isinstance(metrics.box.p, np.ndarray) else float(metrics.box.p.mean())
    recall_value = float(metrics.box.r) if not isinstance(metrics.box.r, np.ndarray) else float(metrics.box.r.mean())

    print("\n📊 总体性能指标:")
    print(f"  • mAP50 (主要指标):     {metrics.box.map50:.4f}")
    print(f"  • mAP50-95 (严格指标):  {metrics.box.map:.4f}")
    print(f"  • 精确度(Precision):    {precision_value:.4f}")
    print(f"  • 召回率(Recall):       {recall_value:.4f}")

    print("\n📈 各类别详细性能:")
    print("-" * 50)
    print(f"{'类别':<15} {'mAP50':<10} {'精确度':<10} {'召回率':<10}")
    print("-" * 50)

    # 获取类别名称
    if hasattr(metrics, 'names'):
        class_names = metrics.names
    else:
        class_names = ['no_helmet', 'helmet']

    # 获取每个类别的指标
    for i, name in enumerate(class_names):
        # 新版本中指标可能是数组
        ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        precision = metrics.box.p[i] if isinstance(metrics.box.p, np.ndarray) and i < len(
            metrics.box.p) else precision_value
        recall = metrics.box.r[i] if isinstance(metrics.box.r, np.ndarray) and i < len(metrics.box.r) else recall_value

        print(f"{name:<15} {ap50:.4f}     {precision:.4f}     {recall:.4f}")

    # 4. 将指标保存为CSV
    results_dict = {
        "模型名称": ["YOLOv11s-安全帽检测"],
        "数据集": ["测试集(945张图片)"],
        "mAP50": [round(metrics.box.map50, 4)],
        "mAP50-95": [round(metrics.box.map, 4)],
        "Precision": [round(precision_value, 4)],
        "Recall": [round(recall_value, 4)],
        "参数数量": ["942万"],
        "输入尺寸": ["640×640"]
    }

    df = pd.DataFrame(results_dict)
    csv_path = "final_test_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"✅ 评估完成！")
    print(f"📁 详细结果保存在: runs/detect/final_test_evaluation")
    print(f"📄 指标表格已保存: {csv_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()