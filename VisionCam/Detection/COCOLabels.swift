import Foundation

/// COCO 80 类的中英文名称。
///
/// 模型（YOLOv8n，COCO 预训练）输出的类别索引就是这张表的顺序，
/// 索引与 CoreML 模型内置的 `stringClassLabels` 完全一致。
enum COCOLabels {

    static let english: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]

    static let chinese: [String] = [
        "人物", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船",
        "红绿灯", "消防栓", "停车标志", "停车计时器", "长椅", "鸟", "猫",
        "狗", "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包",
        "雨伞", "手提包", "领带", "行李箱", "飞盘", "双板滑雪", "单板滑雪", "球",
        "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍",
        "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果",
        "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子",
        "沙发", "盆栽", "床", "餐桌", "马桶", "电视", "笔记本电脑", "鼠标",
        "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽",
        "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机",
        "牙刷",
    ]

    /// `person` 在 COCO 中的类别索引。
    static let personIndex = 0

    /// 按类别索引取中文名，越界时回退成英文名或索引。
    static func displayName(index: Int) -> String {
        if index >= 0 && index < chinese.count {
            return chinese[index]
        }
        if index >= 0 && index < english.count {
            return english[index]
        }
        return "类别 \(index)"
    }
}
