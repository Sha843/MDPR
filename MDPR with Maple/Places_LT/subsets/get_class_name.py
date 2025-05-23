import os

def extract_categories(file_list):
    categories = set()
    for file_path in file_list:
        # 分割路径获取类别目录（适用于路径格式：val/类别名/文件名）
        parts = os.path.normpath(file_path).split(os.sep)
        if len(parts) > 1:
            categories.add(parts[1])  # 假设类别是路径的第二级目录
    return sorted(list(categories))

# 示例用法（需要替换实际数据读取方式）
with open("./Places_LT_test.txt", "r") as f:
    lines = f.readlines()

file_paths = [line.split()[0] for line in lines]  # 分割每行的路径部分
category_list = extract_categories(file_paths)

print(category_list)