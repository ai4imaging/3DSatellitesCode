from PIL import Image
import os
from natsort import natsorted

def png_to_gif(input_folder, output_file, duration=1000):
    """
    将指定文件夹中的所有PNG图片转换为GIF，每帧间隔1秒（1000ms）。

    参数：
    input_folder (str): 包含PNG图片的文件夹路径
    output_file (str): 输出GIF文件的路径
    duration (int): 每帧的持续时间（单位：毫秒，默认值为1000ms）
    """
    # 获取文件夹中的所有PNG文件
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png') and f.startswith('refine')]

    # 使用 natsorted 对文件名进行自然排序
    png_files = natsorted(png_files, reverse=True)




    # 加载图片并存储在列表中
    images = []
    for png_file in png_files:
        image_path = os.path.join(input_folder, png_file)
        img = Image.open(image_path)
        images.append(img)

    # 保存为GIF格式
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0)


# 示例调用
input_folder = 'media'  # 替换为PNG图片所在的文件夹路径
output_file = 'output.gif'  # 输出GIF文件的路径
png_to_gif(input_folder, output_file, duration=1000)
