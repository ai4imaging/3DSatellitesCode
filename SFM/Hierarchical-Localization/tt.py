# 给定参数
focal_length_mm = 3910  # 焦距，单位：毫米
pixel_size_micrometers = 2  # 像素尺寸，单位：微米

# 将像素尺寸转换为毫米
pixel_size_mm = pixel_size_micrometers / 1000  # 2微米 = 0.002毫米

# 计算焦距，单位为像素
focal_length_pixels = focal_length_mm / pixel_size_mm

# 输出结果
print(f"望远镜的焦距为：{focal_length_pixels:.2f} 像素")
