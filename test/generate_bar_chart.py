"""
读取123.xlsx文件并生成交互式堆叠柱状图（HTML格式）
横坐标：价格区间 (<50, 50-100, 100-200, >200)
纵坐标：2023年占比（百分比）
每个价格区间内按功效类别（保湿、美白、抗衰老等）用不同颜色堆叠显示
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os


def read_and_process_data(file_path):
    """读取Excel文件并处理数据"""
    df = pd.read_excel(file_path)

    # 打印原始数据结构
    print("原始数据列名:", df.columns.tolist())
    print("原始数据前几行:")
    print(df.head(10))

    # 数据列: 功效类别, 价格区间（元/片）, 2023年占比
    effect_col = df.columns[0]  # 功效类别
    price_col = df.columns[1]   # 价格区间（元/片）
    value_col = df.columns[2]   # 2023年占比

    print(f"\n使用列: 功效='{effect_col}', 价格='{price_col}', 值='{value_col}'")

    # 标准化价格区间标签（将中文符号转换为英文符号）
    def normalize_price_range(price):
        price_str = str(price)
        # 处理全角/半角小于号和大于号
        price_str = price_str.replace('＜', '<').replace('＜', '<')
        price_str = price_str.replace('＞', '>').replace('＞', '>')
        # 处理不同的破折号
        price_str = price_str.replace('–', '-').replace('—', '-')
        return price_str

    df['标准化价格区间'] = df[price_col].apply(normalize_price_range)

    # 确保价格区间按顺序排列
    price_order = ['<50', '50-100', '100-200', '>200']
    df['标准化价格区间'] = pd.Categorical(df['标准化价格区间'], categories=price_order, ordered=True)
    df = df.sort_values('标准化价格区间')

    return df, effect_col, value_col


def create_stacked_bar_chart_html(data, effect_col, value_col, output_path='/Users/shiyiliu/workspace/pyproject/test/bar_chart.html'):
    """创建交互式堆叠柱状图（HTML格式）"""

    # 定义美妆行业优雅配色方案
    color_map = {
        '保湿': '#A8D8EA',        # 清新蓝 - 保湿水润感
        '美白': '#FFB6C1',        # 樱花粉 - 美白嫩肤感
        '抗衰老': '#D4A5A5',      # 玫瑰棕 - 高级抗老感
        '舒缓修护': '#98D8C8',    # 薄荷绿 - 舒缓镇静感
        '祛痘': '#F7CAC9',        # 柔和粉 - 祛痘修护感
        '清洁': '#B4E7CE',        # 清新绿 - 清洁净透感
        '抗氧化': '#FF9AA2',      # 蜜桃粉 - 抗氧活力感
        '屏障修护': '#C7CEEA',    # 薰衣草紫 - 屏障修护感
        '紧致提拉': '#E2F0CB',    # 嫩芽绿 - 紧致提升感
    }

    # 创建图表
    fig = go.Figure()

    # 获取所有功效类别
    categories = data[effect_col].unique()

    # 为每个功效类别添加一个堆叠柱
    for category in categories:
        category_data = data[data[effect_col] == category]

        fig.add_trace(go.Bar(
            name=category,
            x=category_data['标准化价格区间'],
            y=category_data[value_col],
            marker_color=color_map.get(category, px.colors.qualitative.Set3[len(color_map) % 12]),
            text=category_data.apply(lambda row: f'{category}<br>{row[value_col]:.1%}', axis=1),
            textposition='inside',
            textfont={'size': 12, 'color': '#333', 'family': 'Arial, sans-serif'},
            hovertemplate=f'<b>{category}</b><br>' +
                         '价格区间: %{x}<br>' +
                         '占比: %{y:.1%}<br>' +
                         '<extra></extra>',
        ))

    # 更新布局
    fig.update_layout(
        title={
            'text': '不同价格区间的功效类别分布（2023年）',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='价格区间（元/片）',
        yaxis_title='2023年占比',
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#ddd',
            borderwidth=1
        ),
        plot_bgcolor='#FFFEF9',      # 温暖的米白色背景
        paper_bgcolor='#FFFEF9',     # 整体使用温暖的米白色
        font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        margin=dict(l=60, r=200, t=60, b=60),
        height=600,
        width=1000,
    )

    # 更新坐标轴样式 - 柔和风格
    fig.update_xaxes(
        tickfont=dict(size=14, color='#666'),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.06)',
        linecolor='#E8E8E8',
        linewidth=2
    )
    fig.update_yaxes(
        tickfont=dict(size=14, color='#666'),
        tickformat='.1%',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.06)',
        linecolor='#E8E8E8',
        linewidth=2
    )

    # 保存为HTML文件
    fig.write_html(output_path, config={'displayModeBar': True, 'displaylogo': False})
    print(f"\n图表已保存至: {output_path}")

    return fig


def main():
    """主函数"""
    file_path = '/Users/shiyiliu/workspace/pyproject/test/123.xlsx'

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return

    try:
        # 读取并处理数据
        print("正在读取Excel文件...")
        data, effect_col, value_col = read_and_process_data(file_path)

        # 创建图表
        print("\n正在生成交互式柱状图...")
        create_stacked_bar_chart_html(data, effect_col, value_col)

        print("\n完成! 请在浏览器中打开 bar_chart.html 查看图表。")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
