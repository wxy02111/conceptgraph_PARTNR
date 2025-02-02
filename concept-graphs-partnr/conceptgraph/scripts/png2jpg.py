from PIL import Image
import os

def convert_png_to_jpg_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                print(f"跳过: {file_path} 不是文件")
                continue
            
            # 获取文件名（不含扩展名）
            base_name, _ = os.path.splitext(file_name)
            
            # 打开PNG文件并转换为RGB模式
            try:
                with Image.open(file_path) as img:
                    rgb_img = img.convert('RGB')
                    jpg_path = os.path.join(folder_path, f"{base_name}.jpg")
                    rgb_img.save(jpg_path, 'JPEG')
                    print(f"转换成功: {jpg_path}")
            except Exception as e:
                print(f"转换失败 {file_name}: {e}")

if __name__ == "__main__":
    folder_path = "/home/wxy/SCAI_lab/partnr-planner/data/traj0/agent0/rgb"  # 在此处定义文件夹路径
    convert_png_to_jpg_in_folder(folder_path)