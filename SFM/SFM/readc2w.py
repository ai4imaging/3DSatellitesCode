import json
import numpy as np


def load_cameras_from_json(json_file_path):
    """
    读取JSON文件并返回c2w矩阵的列表。

    参数:
        json_file_path (str): JSON文件的路径。

    返回:
        List[np.ndarray]: 包含每个相机的4x4 c2w矩阵的列表。
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    c2w_list = []
    for camera in data:
        rotation = camera.get('rotation')
        position = camera.get('position')

        if rotation is None or position is None:
            print(f"Camera ID {camera.get('id')} 缺少旋转或位置数据，跳过。")
            continue

        # 转换旋转矩阵为numpy数组
        rotation_matrix = np.array(rotation)
        if rotation_matrix.shape != (3, 3):
            print(f"Camera ID {camera.get('id')} 的旋转矩阵形状不正确，跳过。")
            continue

        # 转换位置为numpy数组
        position_vector = np.array(position).reshape((3, 1))
        if position_vector.shape != (3, 1):
            print(f"Camera ID {camera.get('id')} 的位置向量形状不正确，跳过。")
            continue

        # 构建4x4的c2w矩阵
        c2w = np.eye(4)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = position_vector.flatten()

        c2w_list.append(c2w)

    return c2w_list


def main():
    # 替换为你的JSON文件路径
    json_file_path = 'cameras.json'

    # 加载c2w矩阵列表
    c2w_matrices = load_cameras_from_json(json_file_path)

    # 打印或处理c2w矩阵
    for idx, c2w in enumerate(c2w_matrices):
        print(f"Camera {idx} c2w matrix:\n{c2w}\n")


if __name__ == "__main__":
    main()
