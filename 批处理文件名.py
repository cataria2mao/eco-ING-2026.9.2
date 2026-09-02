import os
import pandas as pd

# ========== 请在这里配置你的路径 ==========
excel_path = r"C:\Users\你的用户名\Desktop\重命名映射表.xlsx"
folder_path = r"C:\待重命名的文件夹"

# 指定Excel里的列名
col_old = "旧文件名"
col_new = "新文件名"
# =======================================

# 1. 读取Excel
try:
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    print(f"❌ 错误：找不到Excel文件 {excel_path}")
    exit()

# 检查列是否存在
if col_old not in df.columns or col_new not in df.columns:
    print(f"❌ 错误：Excel中找不到 '{col_old}' 或 '{col_new}' 列。")
    print(f"当前列名有：{list(df.columns)}")
    exit()

# 2. 过滤掉空行（防止Excel里有空白行导致报错）
df = df.dropna(subset=[col_old, col_new])

# 3. 开始重命名（先来一轮“试运行”，确认无误后再执行）
print("📋 以下是将要执行的重命名操作（试运行）:")
for index, row in df.iterrows():
    old_name = str(row[col_old]).strip()
    new_name = str(row[col_new]).strip()

    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    # 简单的状态检查
    if not os.path.exists(old_path):
        print(f"  ⚠️  [跳过] 找不到文件：{old_name}")
        continue
    if os.path.exists(new_path):
        print(f"  ⚠️  [跳过] 新文件名已存在：{new_name}")
        continue
    print(f"  ✅ {old_name}  ->  {new_name}")

# 4. 询问用户确认
confirm = input("\n⚠️ 确认执行以上重命名吗？(输入 yes 继续): ")
if confirm.lower() != 'yes':
    print("已取消操作。")
    exit()

# 5. 正式执行
print("\n🚀 开始正式重命名...")
success_count = 0
for index, row in df.iterrows():
    old_name = str(row[col_old]).strip()
    new_name = str(row[col_new]).strip()

    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    # 再次检查（防止试运行和正式运行之间文件被动了）
    if not os.path.exists(old_path):
        print(f"  ❌ 失败：找不到文件 {old_name}")
        continue
    if os.path.exists(new_path):
        print(f"  ❌ 失败：目标 {new_name} 已存在，防止覆盖")
        continue

    try:
        os.rename(old_path, new_path)
        print(f"  ✅ 成功：{old_name} -> {new_name}")
        success_count += 1
    except Exception as e:
        print(f"  ❌ 异常：{old_name} 改名失败，原因：{e}")

print(f"\n🎉 全部完成！成功重命名 {success_count} 个文件。")