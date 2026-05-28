import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import shutil

# ================= 配置区域 =================
# 根目录：VOCdevkit 路径
ROOT_DIR = "/root/IP102/VOCdevkit"

# 子集名称
SPLITS = ["train", "val", "test"]

# 子文件夹名称 (根据您的描述固定)
IMG_FOLDER_NAME = "images"
XML_FOLDER_NAME = "labels"  # 注意：这里存放的是 .xml 文件

# 输出根目录 (将生成 YOLO 格式的标准结构)
OUTPUT_ROOT = "/root/IP102/YOLO_FORMAT"


# ===========================================

def get_image_size(img_path):
    """获取图片宽高"""
    try:
        with Image.open(img_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"❌ 无法读取图片 {img_path}: {e}")
        return None, None


def convert_box(xmin, ymin, xmax, ymax, w, h):
    """核心归一化计算"""
    if w == 0 or h == 0:
        return None

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw = xmax - xmin
    bh = ymax - ymin

    nx = cx / w
    ny = cy / h
    nw = bw / w
    nh = bh / h

    # 截断防止超出 [0, 1] (处理标注误差)
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return f"{nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def process_split(split_name):
    """处理单个子集"""
    # 构建输入路径
    src_img_dir = Path(ROOT_DIR) / split_name / IMG_FOLDER_NAME
    src_xml_dir = Path(ROOT_DIR) / split_name / XML_FOLDER_NAME

    # 构建输出路径
    dst_img_dir = Path(OUTPUT_ROOT) / "images" / split_name
    dst_lbl_dir = Path(OUTPUT_ROOT) / "labels" / split_name

    # 创建输出目录
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_xml_dir.exists():
        print(f"⚠️ 跳过 {split_name}: 找不到目录 {src_xml_dir}")
        return

    xml_files = list(src_xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"⚠️ 跳过 {split_name}: {src_xml_dir} 中没有找到 .xml 文件")
        return

    print(f"🚀 开始处理 {split_name} ({len(xml_files)} 个文件)...")

    success_count = 0
    error_count = 0

    for xml_path in xml_files:
        stem = xml_path.stem  # 文件名不含后缀

        # 1. 寻找对应的图片
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']:
            candidate = src_img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"⚠️ 警告: 未找到 {stem} 对应的图片，生成空标签文件。")
            # 即使没图片，也创建一个空 txt 保持对应关系
            (dst_lbl_dir / f"{stem}.txt").touch()
            error_count += 1
            continue

        # 2. 获取图片尺寸
        w, h = get_image_size(img_path)
        if w is None:
            error_count += 1
            continue

        # 3. 解析 XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            lines = []
            for obj in root.findall('object'):
                name_str = obj.find('name').text

                # 验证类别 ID 是否为数字
                if not name_str.isdigit():
                    print(f"⚠️ 警告: {stem} 中类别 '{name_str}' 不是数字，跳过该框。")
                    continue

                class_id = int(name_str)

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                norm_coords = convert_box(xmin, ymin, xmax, ymax, w, h)
                if norm_coords:
                    lines.append(f"{class_id} {norm_coords}")

            # 4. 写入 YOLO txt
            out_txt = dst_lbl_dir / f"{stem}.txt"
            with open(out_txt, 'w') as f:
                if lines:
                    f.write('\n'.join(lines))

            # 5. 复制图片到输出目录 (方便训练时统一管理)
            shutil.copy(str(img_path), str(dst_img_dir / img_path.name))

            success_count += 1

        except Exception as e:
            print(f"❌ 解析 {xml_path} 失败: {e}")
            error_count += 1

    print(f"✅ {split_name} 完成: 成功 {success_count}, 问题 {error_count}")


def main():
    print(f"📂 输入根目录: {ROOT_DIR}")
    print(f"📂 输出根目录: {OUTPUT_ROOT}")

    if not Path(ROOT_DIR).exists():
        print("❌ 错误: 输入目录不存在！请检查 ROOT_DIR 配置。")
        return

    for split in SPLITS:
        process_split(split)

    print("\n🎉 全部完成！")
    print(f"💡 YOLO 格式数据已保存至: {OUTPUT_ROOT}")
    print("   可直接在 data.yaml 中配置:")
    print(f"   path: {OUTPUT_ROOT}")
    print("   train: images/train")
    print("   val: images/val")


if __name__ == "__main__":
    main()