from docx import Document
import pandas as pd
import argparse
from pathlib import Path

# ============ 数据处理函数 ============
def extract_values(config):
    """从CSV提取指定数据"""
    try:
        df = pd.read_csv(config['file'], encoding=config.get('encoding', 'utf-8'))
    except UnicodeDecodeError:
        # 编码错误时自动尝试其他编码
        df = pd.read_csv(config['file'], encoding='GB18030')

    result = {}
    for row_key, row_spec in config['rows'].items():
        # 筛选行
        if 'filter_col' in row_spec:
            mask = df[row_spec['filter_col']] == row_spec['filter_val']
            if not mask.any():
                raise ValueError(f"文件 {config['file']} 中未找到 {row_spec['filter_col']}={row_spec['filter_val']}")
            row_data = df.loc[mask].iloc[0]
        elif 'index' in row_spec:
            row_data = df.iloc[row_spec['index']]
        else:
            raise ValueError("必须指定 'filter_col' 或 'index'")

        # 提取列（自动处理缺失列）
        result[row_key] = {
            col: row_data[col] if col in df.columns else None
            for col in config['columns']
        }

    return result

def main():
    parser = argparse.ArgumentParser(description='利用动物名录的excel表格生成word版')
    # 输入输出文件参数
    parser.add_argument('--work_dir', default='D:/EcoAgentProject/广西风电鸟类监测',
                        help='工作文件路径')
    parser.add_argument('--input_file1', default='种类组成.csv',
                        help='种类组成的csv文件')
    parser.add_argument('--input_file2', default='df_RP.csv',
                        help='动物名录的excel表格')
    parser.add_argument('--output_file', default='陆生动物报告.docx',
                        help='动物名录word版')

    parser.add_argument('--regional_level', default='广西自治区级',
                        help='省级/区域保护级别列表，逗号分隔或JSON格式，例如 "江苏省级"')

    args = parser.parse_args()

    regional_level = args.regional_level
    work_dir = Path(args.work_dir).resolve()
    file1 = work_dir / args.input_file1
    file2 = work_dir / args.input_file2
    output_path = work_dir / args.output_file

    configs = [
        {
            'name': '种类组成',
            'file': file1,
            'encoding': 'GB18030',
            'rows': {
                '合计': {'filter_col': '纲', 'filter_val': '合计'},
                '两栖纲': {'filter_col': '纲', 'filter_val': '两栖纲'},
                '爬行纲':{'filter_col': '纲', 'filter_val': '爬行纲'},
                '鸟纲': {'filter_col': '纲', 'filter_val': '鸟纲'},
                '哺乳纲': {'filter_col': '纲', 'filter_val': '哺乳纲'}
            },
            'columns': ['目', '科', '种', '东洋种', '古北种', '广布种',
                        '国家一级', '国家二级', regional_level, '东洋种占比', '古北种占比',
                        '广布种占比']
        },
        {
            'name': '居留型',
            'file': file2,
            'encoding': 'utf-8',
            'rows': {
                'RP': {'filter_col': '纲', 'filter_val': '鸟纲'},
                'RP_LABEL': {'filter_col': '纲', 'filter_val': 'label'}
            },
            'columns': ['冬候鸟', '夏候鸟', '旅鸟', '留鸟']
        }
    ]

    # ============ 批量提取所有数据 ============
    extracted = {}
    for config in configs:
        print(config['file'])
        extracted[config['name']] = extract_values(config)
        print(extracted)

    # 便捷访问变量
    hj = extracted['种类组成']['合计']
    lq = extracted['种类组成']['两栖纲']
    px = extracted['种类组成']['爬行纲']
    nl = extracted['种类组成']['鸟纲']
    br = extracted['种类组成']['哺乳纲']
    rp = extracted['居留型']['RP']

    # ============ 生成Word文档 ============
    doc = Document()
    doc.add_heading('动物部分', level=0)

    doc.add_paragraph(
        f"xx区分布的陆生脊椎动物有4纲{hj['目']}目{hj['科']}科{hj['种']}种，"
        f"其中东洋种{hj['东洋种']}种，古北种{hj['古北种']}种，广布种{hj['广布种']}种。"
        f"xx区有国家一级重点保护野生动物{hj['国家一级']}种，"
        f"国家二级重点保护野生动物{hj['国家二级']}种，"
        f"{regional_level}重点保护野生动物{hj[regional_level]}种。"
        f"{hj['种']}种动物在各纲中的种类组成、区系和保护等级具体见下表。"
    )

    doc.add_heading('两栖类现状', level=2)  # 建议用标题样式
    doc.add_paragraph(f"xx内野生两栖动物种类有{lq['目']}目{lq['科']}科{lq['种']}种")
    doc.add_paragraph(
        f"按区系类型分，xx内的{lq['种']}种两栖类可分为x种区系类型，"
        f"东洋种{lq['东洋种']}种，占xx两栖类物种数的{lq['东洋种占比']}；"
        f"古北种{lq['古北种']}种，占xx两栖类物种数的{lq['古北种占比']}："
        f"广布种{lq['广布种']}种，占xx两栖类物种数的{lq['广布种占比']}。"
    )

    doc.add_heading('爬行类现状', level=2)  # 建议用标题样式
    doc.add_paragraph(f"xx内野生爬行动物种类有{px['目']}目{px['科']}科{px['种']}种")
    doc.add_paragraph(
        f"按区系类型分，xx内的{px['种']}种爬行类可分为x种区系类型，"
        f"东洋种{px['东洋种']}种，占xx爬行类物种数的{px['东洋种占比']}；"
        f"古北种{px['古北种']}种，占xx爬行类物种数的{px['古北种占比']}；"
        f"广布种{px['广布种']}种，占xx爬行类物种数的{px['广布种占比']}。"
    )

    doc.add_heading('鸟类现状', level=2)  # 建议用标题样式
    doc.add_paragraph(f"xx内野生鸟类有{nl['目']}目{nl['科']}科{nl['种']}种")
    doc.add_paragraph(
        f"按区系类型分，xx内的{nl['种']}种鸟类可分为x种区系类型，"
        f"东洋种{nl['东洋种']}种，占xx鸟类物种数的{nl['东洋种占比']}；"
        f"古北种{nl['古北种']}种，占xx鸟类物种数的{nl['古北种占比']}；"
        f"广布种{nl['广布种']}种，占xx鸟类物种数的{nl['广布种占比']}"
    )

    doc.add_heading('哺乳类现状', level=2)  # 建议用标题样式
    doc.add_paragraph(f"xx内野生哺乳类有{br['目']}目{br['科']}科{br['种']}种")
    doc.add_paragraph(
        f"按区系类型分，xx内的{br['种']}种哺乳类可分为x种区系类型，"
        f"东洋种{br['东洋种']}种，占xx哺乳类物种数的{br['东洋种占比']}；"
        f"古北种{br['古北种']}种，占xx哺乳类物种数的{br['古北种占比']}；"
        f"广布种{br['广布种']}种，占xx哺乳类物种数的{br['广布种占比']}。"
    )

    # 添加鸟类数据（示例）
    doc.add_heading('鸟类现状', level=2)
    doc.add_paragraph(
        f"xx区鸟类中，冬候鸟{rp['冬候鸟']}种，夏候鸟{rp['夏候鸟']}种，旅鸟{rp['旅鸟']}种，留鸟{rp['留鸟']}种。"
    )

    doc.save(output_path)
    print("文档生成完成！")

if __name__ == '__main__':
    main()