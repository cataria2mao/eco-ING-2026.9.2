library(dplyr, warn.conflicts = FALSE)
library(officer)
library(readxl, warn.conflicts = FALSE)
library(tidyr)
library(tibble)
library(scales)
library(ggplot2)
library(optparse)
library(rvg)

windowsFonts(SimSun=windowsFont("SimSun"))

#接受来自LLM的参数
args <- commandArgs(trailingOnly = TRUE)

option_list <- list(
  make_option(c("-o", "--work_dir"), type="character", default="D:/EcoAgentProject/广西风电鸟类监测"),
  make_option(c("-i", "--input_file"),  type="character", default="step1_result.xlsx"),
  make_option(c("-r", "--regional_level"), type="character", default="广西自治区级")
)
opt <- parse_args(OptionParser(option_list=option_list))

work_dir <- opt$work_dir
input_folder <- file.path(work_dir, opt$input_file)
sj  <- opt$regional_level

if (!dir.exists(work_dir)) {
  dir.create(work_dir, recursive = TRUE)
}

# 读取数据
myd <- read_excel(input_folder, sheet = 1)

# 计算纲级分类统计
group_count <- myd %>%
  group_by(纲) %>%
  summarise(
    目 = n_distinct(目),
    科 = n_distinct(科),
    种 = n_distinct(中文名),
    .groups = 'drop'
  )

# 鸟类科级统计（含序号）
bird_stats <- myd %>%
  filter(纲 == "鸟纲") %>%
  group_by(目, 科) %>%
  summarise(
    种 = n_distinct(中文名),
    序号 = first(序号),  # 取科内第一个序号
    .groups = 'drop'
  ) %>%
  select(序号, 目, 科, 种)

#鸟类目级统计和图
bird_stats_order <-bird_stats %>%
  group_by(目) %>%
  summarise(
    种 = sum(种),
    .groups = 'drop'
  )

p1 <- ggplot(bird_stats_order, aes(x=reorder(目, -种),y=种))+
  geom_bar(stat="identity",width=0.3,
           fill = "steelblue",
           colour = "steelblue")+
  geom_text(aes(label = 种), vjust = -0.5, family = "serif") +
  labs( x='', y = "物种数")+
  scale_y_continuous(limits = c(0, max(bird_stats_order[, "种"])+10),expand = c(0, 0))+
  theme_bw()+
  theme(axis.text.x = element_text(size = 14, angle=45, family = "SimSun", 
                                   hjust = 1, face = "bold"), 
        axis.text.y = element_text(size = 12, family = "serif"),
        axis.title = element_text(size = 15, family = "SimSun", face = "bold"))

# 计算鸟纲的居留型比例
df_RP <- myd %>%
  filter(纲 == '鸟纲') %>%
  group_by(居留型) %>%
  summarise(鸟纲 = n()) %>%
  mutate(ratio = 鸟纲 / sum(鸟纲),
         label = paste0(sprintf("%.2f", ratio *100), "%")) %>%
  t() %>%
  as.data.frame() %>%
  setNames(.[1, ]) %>%
  slice(-1) %>%
  rownames_to_column('纲')

# 区系类型统计
region_stats <- myd %>%
  count(纲, 区系类型, name = "count") %>%
  pivot_wider(
    names_from = 区系类型,
    values_from = count,
    values_fill = 0
  )

if (!"东洋种" %in% names(region_stats)) {
  region_stats <- region_stats %>% mutate(东洋种 = 0)
}
if (!"古北种" %in% names(region_stats)) {
  region_stats <- region_stats %>% mutate(古北种 = 0)
}
if (!'广布种' %in% names(region_stats)) {
  region_stats <- region_stats %>% mutate(!!sym(sj) := 0)
}

region_stats <- region_stats %>%
  select(纲, 东洋种, 古北种, 广布种)

# 保护等级统计
protect_stats <- myd %>%
  count(纲, 保护级别, name = "count") %>%
  pivot_wider(
    names_from = 保护级别,
    values_from = count,
    values_fill = 0
  )

if (!"国家一级" %in% names(protect_stats)) {
  protect_stats <- protect_stats %>% mutate(国家一级 = 0)
}
if (!"国家二级" %in% names(protect_stats)) {
  protect_stats <- protect_stats %>% mutate(国家二级 = 0)
}
if (!sj %in% names(protect_stats)) {
  protect_stats <- protect_stats %>% mutate(!!sym(sj) := 0)
}

protect_stats <- protect_stats %>%
  select(纲, 国家一级, 国家二级, !!sym(sj))

# 合并并添加合计行
final_stats <- group_count %>%
  left_join(region_stats, by = "纲") %>%
  left_join(protect_stats, by = "纲")

final_stats <-final_stats %>%
  bind_rows(
    final_stats %>% 
      summarise(
        纲 = "合计",
        across(where(is.numeric), ~sum(., na.rm = TRUE)),
        across(where(is.character), ~"")
      )
  )

final_stats[nrow(final_stats),1] <-"合计"

# 计算区系比例    
final_stats <- final_stats %>%
  mutate(
    东洋种占比 = percent(东洋种 / (东洋种 + 古北种 + 广布种), accuracy = 0.01),
    古北种占比 = percent(古北种 / (东洋种 + 古北种 + 广布种), accuracy = 0.01),
    广布种占比 = percent(广布种 / (东洋种 + 古北种 + 广布种), accuracy = 0.01),
    .after = all_of(sj)
  )

# 因子排序输出
result <- final_stats %>%
  mutate(纲 = factor(纲, 
                    levels = c("两栖纲","爬行纲","鸟纲","哺乳纲","合计"),
                    ordered = TRUE)) %>%
  arrange(纲) %>%
  mutate(纲 = as.character(纲))

# 输出结果
write.csv(result, file.path(work_dir, '种类组成.csv'), fileEncoding = 'GB18030', row.names = FALSE)
write.csv(bird_stats, file.path(work_dir, '鸟类组成.csv'), fileEncoding = 'GB18030', row.names = FALSE)
write.csv(df_RP, file.path(work_dir, 'df_RP.csv'), fileEncoding = 'GB18030', row.names = FALSE)

ppt <- read_pptx()

# 添加一个幻灯片
ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
ppt <- ph_with(ppt, value = dml(ggobj = p1), location = ph_location_fullsize())

# 保存PPT文件
output_ppt <- file.path(work_dir, "niaolei.pptx")
print(ppt, target = output_ppt)


summary <- paste0(
  "【分析完成】\n",
  "输出文件：\n",
  "1. 种类组成：", file.path(work_dir, "种类组成.csv"), "\n",
  "2. 鸟类组成：", file.path(work_dir, "鸟类组成.csv"), "\n",
  "3. 鸟类图表：", file.path(work_dir, "niaolei.pptx"), "\n"
)
cat(summary)
