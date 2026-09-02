#!/usr/bin/env python3
"""
动物列表合并脚本
功能：合并样线表和历史资料中的动物列表，去重并优先保留样线表数据
"""

import argparse
from pathlib import Path
import pandas as pd


def merge_animal_list(work_dir: str, sample_file: str, history_file: str, output_file: str) -> str:
    """
    合并样线表和历史资料中的动物列表

    Args:
        work_dir: 工作目录路径
        sample_file: 样线表文件名（含.xlsx后缀）
        history_file: 历史资料文件名（含.xlsx后缀）
        output_file: 输出文件名（含.xlsx后缀）

    Returns:
        执行结果信息
    """
    # 构建完整路径（使用 pathlib）
    work_path = Path(work_dir)
    sample_path = work_path / sample_file
    history_path = work_path / history_file
    output_path = work_path / output_file

    # 检查文件是否存在
    if not sample_path.exists():
        return f"错误：样线表文件不存在: {sample_path}"
    if not history_path.exists():
        return f"错误：历史资料文件不存在: {history_path}"

    try:
        # 读取样线表
        df_sample = pd.read_excel(sample_path)

        # 检查"中文名"列是否存在
        if "中文名" not in df_sample.columns:
            return f"错误：样线表 {sample_file} 中未找到'中文名'列，现有列: {list(df_sample.columns)}"

        # 提取"中文名"列，添加"来源"列
        df_sample_result = pd.DataFrame()
        df_sample_result["中文名"] = df_sample["中文名"]
        df_sample_result["来源"] = "现场调查"

        # 读取历史资料
        df_history = pd.read_excel(history_path)

        # 检查必要列
        if "中文名" not in df_history.columns:
            return f"错误：历史资料 {history_file} 中未找到'中文名'列，现有列: {list(df_history.columns)}"
        if "来源" not in df_history.columns:
            return f"错误：历史资料 {history_file} 中未找到'来源'列，现有列: {list(df_history.columns)}"

        # 提取"中文名"和"来源"列
        df_history_result = pd.DataFrame()
        df_history_result["中文名"] = df_history["中文名"]
        df_history_result["来源"] = df_history["来源"]

        # 上下拼接（样线表在上，历史资料在下）
        df_merged = pd.concat([df_sample_result, df_history_result], ignore_index=True)

        # 根据"中文名"去重，保留第一次出现的（即样线表数据优先）
        df_final = df_merged.drop_duplicates(subset=["中文名"], keep="first")

        # 重置索引
        df_final = df_final.reset_index(drop=True)

        # 写入输出文件
        df_final.to_excel(output_path, index=False, engine="openpyxl")

        result_info = (
            f"✅ 动物列表合并完成！\n"
            f"   样线表原始记录: {len(df_sample)} 条\n"
            f"   历史资料原始记录: {len(df_history)} 条\n"
            f"   合并后总记录: {len(df_merged)} 条\n"
            f"   去重后记录: {len(df_final)} 条\n"
            f"   去重数量: {len(df_merged) - len(df_final)} 条\n"
            f"   输出文件: {output_path}"
        )
        return result_info

    except Exception as e:
        return f"错误：处理过程中发生异常: {e}"


def main():
    parser = argparse.ArgumentParser(description="动物列表合并脚本")
    parser.add_argument("--work_dir", required=True, help="工作目录路径")
    parser.add_argument("--input_file", required=True, help="样线表文件名（如：样线表.xlsx）")
    parser.add_argument("--history_file", required=True, help="历史资料文件名（如：历史资料.xlsx）")
    parser.add_argument("--output_file", default="动物列表.xlsx", help="输出文件名（如：动物列表.xlsx）")

    args = parser.parse_args()

    result = merge_animal_list(
        work_dir=args.work_dir,
        sample_file=args.input_file,
        history_file=args.history_file,
        output_file=args.output_file
    )

    print(result)
    return result


if __name__ == "__main__":
    main()