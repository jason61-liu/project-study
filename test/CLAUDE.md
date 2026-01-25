# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python data visualization script that reads Excel files containing beauty/skincare product data and generates interactive stacked bar charts. The chart displays the distribution of product efficacy categories (moisturizing, whitening, anti-aging, etc.) across different price ranges.

## Common Commands

**IMPORTANT**: This project uses `uv` for dependency management. You must activate the virtual environment first:

```bash
source ~/workspace/pyproject/.venv/bin/activate
```

### Running the script
```bash
# After activating the virtual environment
python generate_bar_chart.py
```

### Installing dependencies
```bash
# After activating the virtual environment
pip install pandas plotly openpyxl
```

## Architecture

### Single-file structure
The entire application is contained in `generate_bar_chart.py` (~170 lines).

### Data flow
```
123.xlsx (input) → pandas read → normalize price ranges → Plotly visualization → bar_chart.html (output)
```

### Key components

**`read_and_process_data(file_path)`**
- Reads Excel file using pandas
- Normalizes price range labels (handles Chinese/full-width characters: `＜` → `<`, `＞` → `>`)
- Sorts data into defined price order: `<50`, `50-100`, `100-200`, `>200`
- Returns processed DataFrame and column names

**`create_stacked_bar_chart_html(data, effect_col, value_col, output_path)`**
- Creates interactive stacked bar chart using Plotly Graph Objects
- Uses custom beauty industry color palette (9 efficacy categories)
- Outputs self-contained HTML file with interactive hover tooltips
- Hardcoded output path: `/Users/shiyiliu/workspace/pyproject/test/bar_chart.html`

### Hardcoded paths
The script contains hardcoded absolute paths:
- Input: `/Users/shiyiliu/workspace/pyproject/test/123.xlsx`
- Output: `/Users/shiyiliu/workspace/pyproject/test/bar_chart.html`

These must be updated when working in different environments or with different files.

### Color scheme
The chart uses a carefully chosen pastel color palette for beauty industry aesthetics:
- 保湿 (Moisturizing): `#A8D8EA` (fresh blue)
- 美白 (Whitening): `#FFB6C1` (cherry pink)
- 抗衰老 (Anti-aging): `#D4A5A5` (rose brown)
- And 6 other efficacy categories with specific colors
