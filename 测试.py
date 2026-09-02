import subprocess
result = subprocess.run(['Rscript.exe', 'skills/animal_catalog_analysis.R', '--work_dir', 'D:\\EcoAgentProject\\广西风电鸟类监测', '--input_file', '动物名录.xlsx'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=120)
print(result.stdout)
