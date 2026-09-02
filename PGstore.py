import pandas as pd
from sqlalchemy import create_engine

# --- 1. 配置数据库连接 ---
# 数据库连接字符串格式: postgresql://用户名:密码@主机:端口/数据库名
# 请将下面的 'your_username', 'your_password', 'your_database' 替换为你的实际信息
DB_USER = 'animal_agent'
DB_PASSWORD = '010405'  # 替换为你的密码
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'animal_base_data'   # 你要导入的目标数据库

# 创建数据库引擎
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# --- 2. 读取 Excel 文件 ---
# 请将 '参考文件.xlsx' 替换为你的文件实际路径
excel_file_path = 'D:/PythonProject1/skills/参考名录.xlsx'
df = pd.read_excel(excel_file_path, engine='openpyxl')

# (可选) 如果列名需要与数据库表字段匹配，可以在此处重命名
# df.columns = ['col1', 'col2', 'col3']

# --- 3. 导入数据到 PostgreSQL ---
# 定义在数据库中要创建的表名
table_name = 'terrestrial_vertebrates_basic_data'  # 请替换为你想要的表名

# 使用 pandas 的 to_sql 方法将数据写入数据库
# if_exists='replace' 表示如果表存在则替换（先删除再创建）
# if_exists='append' 表示如果表存在则追加数据
# index=False 表示不将 DataFrame 的索引作为一列写入数据库
df.to_sql(table_name, engine, if_exists='append', index=False)

print(f"数据已成功从 '{excel_file_path}' 导入到数据库 '{DB_NAME}' 的表 '{table_name}' 中。")


