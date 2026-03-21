#!/usr/bin/env python3
"""
南山初中名校录取率数据分析脚本 - 生成PDF报告
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

# 设置中文字体 - 尝试多个中文字体
fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'SimHei', 'Heiti TC', 'Microsoft YaHei']
available_fonts = [f.name for f in font_manager.fontManager.ttflist]
chinese_font = None
for font in fonts:
    if font in available_fonts:
        chinese_font = font
        break

if chinese_font:
    matplotlib.rcParams['font.sans-serif'] = [chinese_font]
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f"使用中文字体: {chinese_font}")
else:
    print("警告: 未找到合适的中文字体，图表可能无法正常显示中文")

# 设置绘图风格
sns.set_style("whitegrid")

# 数据定义
# 南山学校数据
nanshan_data = {
    '年份': [2011, 2010, 2008, 2007],
    '毕业生人数': [253, 279, 233, 195],
    '深圳中学': [5, 10, 6, 8],
    '深圳外国语': [14, 18, 12, 9],
    '深圳实验': [5, 6, 2, 8],
    '深圳高级': [8, 20, 11, 5],
    '四大名校总数': [32, 54, 31, 30],
    '四大名校录取率': [12.65, 19.35, 13.30, 15.38]
}

# 南山实验麒麟部数据
nanshan_experimental_data = {
    '年份': [2011, 2010, 2008, 2007],
    '毕业生人数': [517, 499, 475, 387],
    '深圳中学': [19, 15, 13, 7],
    '深圳外国语': [31, 20, 22, 24],
    '深圳实验': [36, 44, 15, 15],
    '深圳高级': [33, 32, 30, 41],
    '四大名校总数': [119, 111, 80, 87],
    '四大名校录取率': [23.02, 22.24, 16.84, 22.48]
}

# 创建DataFrame
df_nanshan = pd.DataFrame(nanshan_data)
df_nanshan_experimental = pd.DataFrame(nanshan_experimental_data)

# 添加学校标识
df_nanshan['学校'] = '南山学校'
df_nanshan_experimental['学校'] = '南山实验麒麟部'

# 输出文件路径
excel_file = '/Users/shiyiliu/workspace/pyproject/summary-wechat/nanshan_admission_data.xlsx'
pdf_file = '/Users/shiyiliu/workspace/pyproject/summary-wechat/nanshan_admission_report.pdf'

# 保存到Excel
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    df_nanshan.to_excel(writer, sheet_name='南山学校', index=False)
    df_nanshan_experimental.to_excel(writer, sheet_name='南山实验麒麟部', index=False)
    df_combined = pd.concat([df_nanshan, df_nanshan_experimental], ignore_index=True)
    df_combined.to_excel(writer, sheet_name='汇总', index=False)

print(f"Excel数据已保存到: {excel_file}")

# 创建PDF报告
print("正在生成PDF报告...")

# 数据分析
avg_nanshan = df_nanshan['四大名校录取率'].mean()
avg_experimental = df_nanshan_experimental['四大名校录取率'].mean()

# 创建PDF
with PdfPages(pdf_file) as pdf:
    # === 封面页 ===
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.7, '南山初中名校录取率分析报告',
            fontsize=32, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#2c3e50')

    ax.text(0.5, 0.55, '2007-2011年数据统计分析',
            fontsize=18, ha='center', va='center',
            transform=ax.transAxes, color='#7f8c8d')

    # 统计摘要
    summary_text = f"""
    数据来源: https://max.book118.com/html/2024/0825/7122014013006145.shtm

    分析范围:
    • 南山学校
    • 南山实验麒麟部

    目标高中:
    • 深圳中学
    • 深圳外国语学校
    • 深圳实验学校
    • 深圳高级中学

    报告生成时间: {pd.Timestamp.now().strftime('%Y年%m月%d日')}
    """

    ax.text(0.5, 0.35, summary_text,
            fontsize=12, ha='center', va='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 封面页已生成")

    # === 数据表格页 ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('数据详情', fontsize=20, fontweight='bold', y=0.98)

    # 南山学校表格
    ax1.axis('off')
    ax1.set_title('南山学校', fontsize=14, fontweight='bold', pad=10)

    table_data1 = []
    table_data1.append(['年份', '毕业生', '深中', '深外', '深实', '深高', '总数', '录取率'])
    for i, row in df_nanshan.iterrows():
        table_data1.append([
            str(int(row['年份'])),
            str(int(row['毕业生人数'])),
            str(int(row['深圳中学'])),
            str(int(row['深圳外国语'])),
            str(int(row['深圳实验'])),
            str(int(row['深圳高级'])),
            str(int(row['四大名校总数'])),
            f"{row['四大名校录取率']:.2f}%"
        ])

    table1 = ax1.table(cellText=table_data1, cellLoc='center', loc='center',
                       colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)

    # 设置表头样式
    for i in range(8):
        table1[(0, i)].set_facecolor('#3498db')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    # 南山实验麒麟部表格
    ax2.axis('off')
    ax2.set_title('南山实验麒麟部', fontsize=14, fontweight='bold', pad=10)

    table_data2 = []
    table_data2.append(['年份', '毕业生', '深中', '深外', '深实', '深高', '总数', '录取率'])
    for i, row in df_nanshan_experimental.iterrows():
        table_data2.append([
            str(int(row['年份'])),
            str(int(row['毕业生人数'])),
            str(int(row['深圳中学'])),
            str(int(row['深圳外国语'])),
            str(int(row['深圳实验'])),
            str(int(row['深圳高级'])),
            str(int(row['四大名校总数'])),
            f"{row['四大名校录取率']:.2f}%"
        ])

    table2 = ax2.table(cellText=table_data2, cellLoc='center', loc='center',
                       colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)

    # 设置表头样式
    for i in range(8):
        table2[(0, i)].set_facecolor('#e74c3c')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 数据表格页已生成")

    # === 录取率对比页 ===
    fig, ax = plt.subplots(figsize=(12, 7))

    years = [2011, 2010, 2008, 2007]
    x = np.arange(len(years))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_nanshan['四大名校录取率'], width,
                   label='南山学校', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_nanshan_experimental['四大名校录取率'], width,
                   label='南山实验麒麟部', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('录取率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('四大名校录取率对比', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 30)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)

    # 添加平均线
    ax.axhline(y=avg_nanshan, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=avg_experimental, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)

    # 注释
    ax.text(0.02, 0.98, f'南山学校平均: {avg_nanshan:.2f}%',
            transform=ax.transAxes, fontsize=10, color='#3498db',
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.02, 0.90, f'南山实验平均: {avg_experimental:.2f}%',
            transform=ax.transAxes, fontsize=10, color='#e74c3c',
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 录取率对比页已生成")

    # === 各高中录取趋势页 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('各高中录取人数趋势', fontsize=16, fontweight='bold')

    schools = ['深圳中学', '深圳外国语', '深圳实验', '深圳高级']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'D']

    # 南山学校
    for i, school in enumerate(schools):
        ax1.plot(years, df_nanshan[school].values, marker=markers[i],
                linewidth=2.5, markersize=8, label=school, color=colors[i])
    ax1.set_title('南山学校', fontsize=13, fontweight='bold')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('录取人数')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(years)

    # 南山实验麒麟部
    for i, school in enumerate(schools):
        ax2.plot(years, df_nanshan_experimental[school].values, marker=markers[i],
                linewidth=2.5, markersize=8, label=school, color=colors[i])
    ax2.set_title('南山实验麒麟部', fontsize=13, fontweight='bold')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('录取人数')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(years)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 各高中录取趋势页已生成")

    # === 四大名校总录取人数对比页 ===
    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, df_nanshan['四大名校总数'], width,
                   label='南山学校', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_nanshan_experimental['四大名校总数'], width,
                   label='南山实验麒麟部', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('录取人数', fontsize=12, fontweight='bold')
    ax.set_title('四大名校总录取人数对比', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 总录取人数对比页已生成")

    # === 录取率占比堆叠图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('各高中录取占比分布', fontsize=16, fontweight='bold')

    # 南山学校
    bottom_ns = np.zeros(len(years))
    for i, school in enumerate(schools):
        values = [df_nanshan[df_nanshan['年份'] == y][school].values[0] for y in years]
        percentages = [v/df_nanshan[df_nanshan['年份'] == y]['四大名校总数'].values[0]*100
                      for v, y in zip(values, years)]
        ax1.bar(x, percentages, bottom=bottom_ns, label=school, color=colors[i], alpha=0.8)
        bottom_ns += percentages

    ax1.set_title('南山学校', fontsize=13, fontweight='bold')
    ax1.set_ylabel('占比 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 南山实验麒麟部
    bottom_exp = np.zeros(len(years))
    for i, school in enumerate(schools):
        values = [df_nanshan_experimental[df_nanshan_experimental['年份'] == y][school].values[0] for y in years]
        percentages = [v/df_nanshan_experimental[df_nanshan_experimental['年份'] == y]['四大名校总数'].values[0]*100
                      for v, y in zip(values, years)]
        ax2.bar(x, percentages, bottom=bottom_exp, label=school, color=colors[i], alpha=0.8)
        bottom_exp += percentages

    ax2.set_title('南山实验麒麟部', fontsize=13, fontweight='bold')
    ax2.set_ylabel('占比 (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 录取占比分布页已生成")

    # === 分析结论页 ===
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    ax.text(0.5, 0.92, '分析结论与建议',
            fontsize=24, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#2c3e50')

    conclusions = """
    【一】录取率对比分析

    • 南山实验麒麟部四大名校录取率显著高于南山学校
      - 南山实验麒麟部平均录取率: 21.14%
      - 南山学校平均录取率: 15.17%
      - 差距: 约6个百分点

    • 南山学校在2010年表现最佳(19.35%)，但年度波动较大
    • 南山实验麒麟部整体表现更稳定，录取率基本维持在20%以上


    【二】录取规模分析

    • 南山实验麒麟部毕业生规模约为南山学校的2倍
    • 但四大名校录取人数是南山学校的2.5-3倍
    • 说明南山实验麒麟部在规模效应下仍保持了更高的录取效率


    【三】目标学校特点

    • 深圳外国语和深圳实验是两校主要的录取方向
    • 南山实验在深圳实验学校的录取优势明显
    • 南山学校在2010年深圳高级录取表现突出(20人)


    【四】建议

    1. 南山学校可参考南山实验麒麟部的教学经验
    2. 关注两校在深圳实验学校的录取差异原因
    3. 建议长期追踪数据，分析年度波动因素
    """

    ax.text(0.1, 0.85, conclusions,
            fontsize=11, ha='left', va='top',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
            family='monospace')

    # 添加数据来源说明
    ax.text(0.5, 0.08, f'数据来源: {pdf_file}\n报告生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=9, ha='center', va='bottom',
            transform=ax.transAxes, color='#7f8c8d', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  - 分析结论页已生成")

print(f"\n✅ PDF报告已生成: {pdf_file}")
print(f"   共包含 {pdf.get_pagecount()} 页")

# 打印统计摘要
print("\n" + "="*60)
print("数据统计摘要")
print("="*60)
print("\n【南山学校】")
print(df_nanshan.to_string(index=False))
print(f"\n平均录取率: {avg_nanshan:.2f}%")

print("\n【南山实验麒麟部】")
print(df_nanshan_experimental.to_string(index=False))
print(f"\n平均录取率: {avg_experimental:.2f}%")

print(f"\n录取率差距: {avg_experimental - avg_nanshan:.2f}个百分点")
print("\n分析完成！")
