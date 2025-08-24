import os
import cv2
import pickle

def draw_points_on_images(input_folder, points_list, output_folder, point_color=(0, 0, 255), point_radius=2):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片文件
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持的图片格式
            # 构建图片的完整路径
            image_path = os.path.join(input_folder, filename)
            # 读取图片
            image = cv2.imread(image_path)

            if image is None:
                print(f"无法读取图片: {filename}")
                continue

            # 在图片上绘制特征点
            for point in points_list[i]:
                x, y = point
                cv2.circle(image, (int(x), int(y)), point_radius, point_color, -1)

            # 构建输出路径
            output_path = os.path.join(output_folder, filename)
            # 保存修改后的图片
            cv2.imwrite(output_path, image)
            print(f"已保存图片到: {output_path}")


with open('data_css/keypoints.pkl', 'rb') as file:
    key_points = pickle.load(file)

points_list = []
for i in range(14):
    points_list.append(key_points[i])

# 示例用法
input_folder = 'css_processed'  # 输入文件夹路径
output_folder = 'css_pointed'  # 输出文件夹路径


draw_points_on_images(input_folder, points_list, output_folder)
