import os

# 替换为包含图片的文件夹路径
folder_path = '.'

for filename in os.listdir(folder_path):
    if filename.startswith('frame') and filename.endswith('.jpg'):
        # 提取数字部分并转换成整数，然后格式化为3位数字字符串
        num = int(filename.split('frame')[1].split('.jpg')[0])
        new_filename = f"frame{num:03}.jpg"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f'Renamed {filename} to {new_filename}')
