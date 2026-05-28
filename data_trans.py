import os
import xml.etree.ElementTree as ET
from pathlib import Path

# 配置路径
# 请根据你的实际 IP102 目录结构调整
IMG_DIR = "IP102/images"  # 原始图片目录
ANN_DIR = "IP102/annotations"  # 原始 XML 标注目录
OUTPUT_IMG_DIR = "ip102_yolo/images"
OUTPUT_LABEL_DIR = "ip102_yolo/labels"

# 类别列表 (必须与你的 classes.txt 顺序一致，或者直接在这里定义)
# 这里仅为示例，请替换为 IP102 真实的 102 个类别名
CLASSES = ['riceleafroller','riceleafcaterpillar','paddystemmaggot','asiaticriceborer','yellowriceborer','ricegallmidge','RiceStemfly',
        'brownplanthopper','whitebackedplanthopper','smallbrownplanthopper','ricewaterweevil','riceleafhopper','grainspreaderthrips',
        'riceshellpest','grub','molecricket','wireworm','whitemarginedmoth','blackcutworm','largecutworm','yellowcutworm',
        'redspider','cornborer','armyworm','aphids','Potosiabrevitarsis','peachborer','englishgrainaphid','greenbug',
        'birdcherry-oataphid','wheatblossommidge','penthaleusmajor','longleggedspidermite','wheatphloeothrips','wheatsawfly',
        'cerodontadenticornis','beetfly','fleabeetle','cabbagearmyworm','beetarmyworm','Beetspotflies','meadowmoth','beetweevil',
        'sericaorientalismotschulsky','alfalfaweevil','flaxbudworm','alfalfaplantbug','tarnishedplantbug','Locustoidea','lyttapolita',
        'legumeblisterbeetle','blisterbeetle','therioaphismaculataBuckton','odontothripsloti','Thrips','alfalfaseedchalcid',
        'Pieriscanidia','Apolyguslucorum','Limacodidae','Viteusvitifoliae','Colomerusvitis','BrevipoalpuslewisiMcGregor','oidesdecempunctata',
        'Polyphagotarsonemuslatus','PseudococcuscomstockiKuwana','parathreneregalis','Ampelophaga','Lycormadelicatula','Xylotrechus','Cicadellaviridis','Miridae',
        'Trialeurodesvaporariorum','Erythroneuraapicalis','Papilioxuthus','PanonchuscitriMcGregor','Phyllocoptesoleiverusashmead','IceryapurchasiMaskell',
        'Unaspisyanonensis','Ceroplastesrubens','Chrysomphalusaonidum','ParlatoriazizyphusLucus','Nipaecoccusvastalor','Aleurocanthusspiniferus',
        'TetradacuscBactroceraminax','Dacusdorsalis(Hendel)','Bactroceratsuneonis','Prodenialitura','Adristyrannus','PhyllocnistiscitrellaStainton','Toxopteracitricidus',
        'Toxopteraaurantii','AphiscitricolaVanderGoot','ScirtothripsdorsalisHood','Dasineurasp','LawanaimitataMelichar','SalurnismarginellaGuerr','DeporausmarginatusPascoe',
        'Chlumetiatransversa','Mangoflatbeakleafhopper','Rhytidoderabowriniiwhite','Sternochetusfrigidus','Cicadellidae']


def convert_bbox(size, box):
    """将 VOC (xmin, ymin, xmax, ymax) 转换为 YOLO (x_center, y_center, w, h) 并归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASSES:
            continue  # 跳过不在类别表中的目标

        class_id = CLASSES.index(class_name)

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        objects.append((class_id, xmin, ymin, xmax, ymax))

    # 获取图片尺寸
    size_elem = root.find('size')
    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)

    return objects, (width, height)


def process_dataset(split_name, img_subdir, ann_subdir):
    """处理单个子集 (train/val/test)"""
    src_img_path = Path(IMG_DIR) / img_subdir
    src_ann_path = Path(ANN_DIR) / ann_subdir

    dst_img_path = Path(OUTPUT_IMG_DIR) / split_name
    dst_label_path = Path(OUTPUT_LABEL_DIR) / split_name

    dst_img_path.mkdir(parents=True, exist_ok=True)
    dst_label_path.mkdir(parents=True, exist_ok=True)

    # 遍历标注文件 (假设标注文件和图片一一对应，且后缀匹配)
    # 如果结构不同，请调整遍历逻辑
    for xml_file in src_ann_path.glob("*.xml"):
        img_name = xml_file.stem  # 去掉 .xml 后缀
        # 寻找对应的图片 (可能是 jpg, png, jpeg 等)
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = src_img_path / f"{img_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if not img_file:
            print(f"Warning: Image not found for {xml_file}")
            continue

        # 复制图片
        import shutil
        shutil.copy(str(img_file), str(dst_img_path / img_file.name))

        # 解析 XML 并转换
        try:
            objects, size = parse_xml(xml_file)
            if not objects:
                continue  # 如果没有检测到目标，可以不生成 txt 或生成空 txt

            with open(dst_label_path / f"{img_name}.txt", 'w') as f:
                for cls_id, xmin, ymin, xmax, ymax in objects:
                    x, y, w, h = convert_bbox(size, (xmin, ymin, xmax, ymax))
                    # 过滤掉极小的框或归一化后超出范围的框 (可选)
                    if w <= 0 or h <= 0:
                        continue
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")


# 执行转换
# 假设你的 IP102 目录结构是 images/train, annotations/train 等
# 如果不是，请修改下面的参数
process_dataset("train", "train", "train")
process_dataset("val", "val", "val")
# 如果有 test 集
# process_dataset("test", "test", "test")

print("Conversion completed!")