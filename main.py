from ultralytics import YOLO

if __name__ == '__main__':
    # 训练  
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8.yaml")
    # model.train(data=r"ultralytics/cfg/datasets/mydata.yaml")

    # 验证
    # model = YOLO(r"多模态预训练权重.pt")
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml",batch=1)

    # 检测
    model = YOLO(r"多模态预训练权重.pt")
    model.predict(source=r"datasets/LLVIP700/images/val", save=True)  # 只需要写RGB图片的路径
