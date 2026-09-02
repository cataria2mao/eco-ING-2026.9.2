import sys, argparse, re
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd

def find_all_substring_indices(text: str, substring: str) -> list:
    """返回 substring 在 text 中所有出现位置的起始索引列表"""
    indices = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        indices.append(start)
        start += 1
    return indices

def replace_resident_type(resident):
    if not isinstance(resident, str):
        return resident
    mapping = {'S': '夏候鸟', 'W': '冬候鸟', 'P': '旅鸟', 'R': '留鸟'}
    # 逐个字符替换，非映射字符（如逗号、空格、括号等）原样保留
    return ''.join(mapping.get(ch, ch) for ch in resident)

def remove_lowercase(s):
    s = re.sub(r'[a-z]', '', s)
    s = re.sub(r'[1-5]', '', s)
    return s

def replace_region_stats(region_stats):
    if not isinstance(region_stats, str):
        return region_stats
    mapping = {**dict.fromkeys(['C', 'M', 'U'], '古北种'),
               **dict.fromkeys(['S', 'H', 'W', 'E'], '东洋种'),
               **dict.fromkeys({'O'}, '广布种')}
    # 逐个字符替换，非映射字符（如逗号、空格、括号等）原样保留
    return ''.join(mapping.get(ch, ch) for ch in region_stats)

def main():
    parser = argparse.ArgumentParser(description='对陆生动物物种数据初步整理，输出物种名录。')
    # 输入输出文件参数
    parser.add_argument('--work_dir', default='D:/EcoAgentProject/广西风电鸟类监测',
                        help='工作文件路径')
    parser.add_argument('--input_file', default='动物列表.xlsx',
                        help='待分析文件')
    parser.add_argument('--ref_file', default='参考名录.xlsx',
                        help='参考名录文件路径')
    parser.add_argument('--output_file', default='动物名录.xlsx',
                        help='动物名录excel版')

    # 可选项参数
    parser.add_argument('--pa', default='ⅤA',
                        help='居留型解析的起始标记，如 "ⅤA"')
    parser.add_argument('--regional_level', default='广西自治区级',
                        help='省级/区域保护级别列表，逗号分隔或JSON格式，例如 "四川"')

    args = parser.parse_args()
    regional_level = [args.regional_level]
    print(regional_level)
    work_dir = Path(args.work_dir).resolve()  # resolve() 会统一格式并转为绝对路径
    input_folder = work_dir / args.input_file
    output_folder = work_dir / args.output_file
    script_dir = Path(__file__).parent.resolve()
    ref_path = Path(args.ref_file)
    ref_file = script_dir / ref_path

    if not input_folder.exists():
        print(f"错误：输入文件 {input_folder} 不存在")
        sys.exit(1)
    if not ref_file.exists():
        print(f"错误：参考名录文件 {args.ref_file} 不存在")
        sys.exit(1)

    data = pd.read_excel(input_folder, sheet_name=0)
    #可选本地文件
    #data1 = pd.read_excel(ref_file,sheet_name=0)

    # ---------- 配置数据库连接 ----------
    # 使用之前创建的 animal_agent 用户
    DB_USER = 'animal_agent'
    DB_PASSWORD = '010405'  # 替换为实际密码
    DB_HOST = 'localhost'  # 或 IP 地址
    DB_PORT = '5432'
    DB_NAME = 'animal_base_data'  # 目标数据库

    # 创建 SQLAlchemy 引擎
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # ---------- 读取表数据 ----------
    table_name = 'terrestrial_vertebrates_basic_data'

    # 方法1：使用 read_sql_table（推荐，直接指定表名）
    data1 = pd.read_sql_table(table_name, engine, schema='public')



    data['中文名'] = data['中文名'].astype(str)\
                                  .str.replace(r'[A-Za-z]', '', regex=True)\
                                  .str.strip()

    df_all = pd.merge(data1, data, on='中文名', how='right')

    # 关键词按优先级排好
    national_levels = ['国家一级', '国家二级']

    for idx, row in df_all.iterrows():
        text = row['保护级别']
        if not isinstance(text, str):
            continue

        hit = None
        for lev in national_levels:
            if lev in text:
                hit = lev
                break
        if hit is None:
            for lev in regional_level:
                if lev in text:
                    hit = lev
                    break
        df_all.loc[idx, '保护级别'] = hit

    # 居留型提取（使用传入的 pa）
    df_all['居留型1'] = ""

    # 遍历 DataFrame 的每一行
    for index, row in df_all.iterrows():
        text = row['居留型']
        content = []

        if not isinstance(text, str):
            continue

        starts = find_all_substring_indices(text, args.pa)
        if not starts:
            continue  # 如果未找到起始括号，跳过当前迭代

        for start in starts:
            start += len(f"{args.pa}(")
            end = text.find(")", start)
            if end == -1:
                continue
            content.append(text[start:end])

        df_all.at[index, '居留型1'] = ', '.join(content)  # 将内容写入新列

    df_all['居留型'] = df_all['居留型1']

    df_all = df_all.drop(['居留型1'], axis=1)

    # 替换字母
    df_all['居留型'] = df_all['居留型'].apply(replace_resident_type)

    #地理区系
    df_all['区系类型'] = df_all['分布型'].apply(
        lambda x: remove_lowercase(x) if isinstance(x, str) else x
    )

    for idx, row in df_all.iterrows():
        text = row['区系类型']
        if not isinstance(text, str):
            continue

        text = replace_region_stats(text)
        df_all.loc[idx, '区系类型'] = text

    # 保存结果
    df_all.to_excel(output_folder, sheet_name="Sheet1", index=False, na_rep='')

    print(f"{output_folder}")


if __name__ == "__main__":
    main()
