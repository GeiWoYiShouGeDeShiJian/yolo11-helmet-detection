from ultralytics import YOLO


def main():
    model = YOLO('yolo11s.pt')

    results = model.train(
        data='dataset/data.yaml',
        epochs=50,  # 总目标50轮
        imgsz=640,  # 保持分辨率
        batch=4,  # 关键：将批次大小从8降至4，确保显存充足
        device='0',
        workers=2,  # 关键：降低数据加载线程，减少CUDA冲突风险
        name='helmet_final_phase',  # 使用新名字，避免混淆
        resume=False,  # 设为False，因为我们是加载一个完整的权重重新开始
        lr0=0.001,  # 使用较小的学习率进行微调
        optimizer='AdamW',  # 使用AdamW优化器，通常更稳定
        patience=15,  # 如果连续15轮验证指标没有提升，则提前停止
        amp=True,  # 保持混合精度训练以节省显存
        val=True,  # 每轮都验证
        save=True,
        pretrained=False,  # 关键：设为False，因为我们已经加载了预训练权重
    )
    print("最终阶段训练完成！")


if __name__ == '__main__':
    main()