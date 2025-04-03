import os
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def setup_chinese_font():
    """设置中文字体，优先使用系统自带中文字体"""
    system = platform.system()
    
    if system == 'Windows':  # Windows系统
        font_list = [
            ('Microsoft YaHei', 'msyh.ttc'),  # 微软雅黑
        ]
        
        for font_name, font_file in font_list:
            try:
                font_path = f"C:\\Windows\\Fonts\\{font_file}"
                if os.path.exists(font_path):
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"成功设置中文字体: {font_name}")
                    return FontProperties(fname=font_path)
            except Exception as e:
                print(f"尝试加载字体 {font_name} 失败: {str(e)}")
                continue
                
    elif system == 'Darwin':  # macOS系统
        font_list = [
            '/System/Library/Fonts/PingFang.ttc',           # 苹方
            '/Library/Fonts/Microsoft/Microsoft Sans Serif.ttf',  # 微软雅黑
        ]
        
        for font_path in font_list:
            try:
                if os.path.exists(font_path):
                    font = FontProperties(fname=font_path)
                    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK'] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"成功设置中文字体: {font_path}")
                    return font
            except Exception as e:
                print(f"尝试加载字体失败 {font_path}: {str(e)}")
                continue
                
    else:  # Linux系统
        font_list = [
            'WenQuanYi Micro Hei',
            'Noto Sans CJK JP',
            'Noto Sans CJK SC',
            'Noto Sans CJK TC',
            'Droid Sans Fallback',
            'Source Han Sans CN'
        ]
        
        for font_name in font_list:
            try:
                if font_name in mpl.font_manager.findSystemFonts(fontpaths=None):
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"成功设置中文字体: {font_name}")
                    return FontProperties(family=font_name)
            except Exception as e:
                print(f"尝试加载字体 {font_name} 失败: {str(e)}")
                continue
    
    # 如果所有字体都失败了，尝试使用matplotlib内置的中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print("使用 DejaVu Sans 作为后备字体")
        return FontProperties(family='DejaVu Sans')
    except:
        print("警告: 未找到可用的中文字体，图表中文可能显示为方框")
        return None

def init_plot_style():
    """初始化绘图样式设置"""
    # 初始化中文字体
    chinese_font = setup_chinese_font()
    
    # 确保所有绘图函数都使用中文字体
    mpl.rcParams['font.family'] = plt.rcParams['font.sans-serif'][0]
    mpl.rcParams['axes.unicode_minus'] = False
    
    # 设置全局字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    return chinese_font 