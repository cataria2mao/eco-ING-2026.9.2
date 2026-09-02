library(dplyr, warn.conflicts = FALSE)
library(officer)
library(readxl, warn.conflicts = FALSE)
library(tidyr)
library(reshape2)
library(ggplot2)
library(export)
library(vegan)
library(optparse)
library(rvg)


windowsFonts(SimSun=windowsFont("SimSun"))

# 自定义 Margalef 指数
Margalef.DM<-function(data){
  M<-(length(data)-1)/log(sum(data))
  return(M)
}

#接受来自LLM的参数
args <- commandArgs(trailingOnly = TRUE)

option_list <- list(
  make_option(c("-o", "--work_dir"), type="character", default="D:/EcoAgentProject/广西风电鸟类监测"),
  make_option(c("-i", "--input_file"),  type="character", default="广西风电样线表.xlsx"),
  make_option(c("-g", "--groups_type"),  type="character", default="无")
)

opt <- parse_args(OptionParser(option_list=option_list))

work_dir <- opt$work_dir
input_folder <- file.path(work_dir, opt$input_file)
groups_type <- opt$groups_type


# 读取数据
myd <- read_excel(input_folder, sheet = 1)


if (groups_type == "季节") {
  df <- myd %>%
    group_by(季节, 中文名) %>%
    summarise(nums = sum(数量),
              .groups = "drop")
  
  df <- df%>%
    group_by(季节) %>%
    mutate(优势度 = sprintf("%.2f%%", nums/sum(nums)*100))
  
  jijie <- data.frame(unique(df[,'季节']))
  col1 <- data.frame(matrix(0,length(jijie$季节),4))
  dyxD <- cbind(jijie,col1)
  colnames(dyxD) <- c('jijie', 'Simpson.D', 'Shannon.H','Pielou.E','DM')
  
  for (ji in jijie$季节) {
    vec <- df$nums[df$季节 == ji]          # 先拿向量
    dyxD$Simpson.D[dyxD$jijie == ji] <- diversity(vec, index = "simpson")
    dyxD$Shannon.H[dyxD$jijie == ji] <- diversity(vec, index = "shannon")
    dyxD$Pielou.E[dyxD$jijie == ji] <- diversity(vec, index = "shannon")/log(specnumber(vec))
    dyxD$DM[dyxD$jijie == ji] <- Margalef.DM(vec)
  }
  
  # 将数据转换为长格式（适合ggplot2）
  df_long <- melt(dyxD[, c("jijie", "Simpson.D", "Shannon.H", "Pielou.E")], 
                  id.vars = "jijie", 
                  variable.name = "Index", 
                  value.name = "Value")
  
  # 设置季节顺序（可选，按自然顺序排列）
  df_long$jijie <- factor(df_long$jijie, levels = c("春季", "夏季", "秋季", "冬季"))
  df$jijie <- droplevels(df$jijie)
  
  
  # 分面图
  p <- ggplot(df_long, aes(x = jijie, y = Value, fill = Index)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    facet_wrap(~Index, scales = "free_y", 
               labeller = labeller(Index = c("Simpson.D" = "Simpson指数",
                                             "Shannon.H" = "Shannon指数",
                                             "Pielou.E" = "Pielou指数"))) +
    scale_fill_manual(values = c("Simpson.D" = "#E74C3C", 
                                 "Shannon.H" = "#3498DB", 
                                 "Pielou.E" = "#2ECC71")) +
    labs(title = "不同季节多样性指数",
         x = "季节",
         y = "指数值") +
    theme_bw() +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      strip.background = element_rect(fill = "gray95"),
      strip.text = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10)
    )
  
  ppt <- read_pptx()
  
  # 添加一个幻灯片
  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, value = dml(ggobj = p1), location = ph_location_fullsize())
  
  # 保存PPT文件
  output_ppt <- file.path(work_dir, "diversity_facet_plot.pptx")
  print(ppt, target = output_ppt)
  
}else if (groups_type == "地区") {
  df <- myd %>%
    group_by(地区, 中文名) %>%
    summarise(nums = sum(数量),
              .groups = "drop")
  
  df <- df%>%
    group_by(地区) %>%
    mutate(优势度 = sprintf("%.2f%%", nums/sum(nums)*100))
  
  jijie <- data.frame(unique(df[,"地区"]))
  col1 <- data.frame(matrix(0,length(jijie$地区),4))
  dyxD <- cbind(jijie,col1)
  colnames(dyxD) <- c('jijie', 'Simpson.D', 'Shannon.H','Pielou.E','DM')
  
  for (ji in jijie$地区) {
    vec <- df$nums[df$地区 == ji]          # 先拿向量
    dyxD$Simpson.D[dyxD$jijie == ji] <- diversity(vec, index = "simpson")
    dyxD$Shannon.H[dyxD$jijie == ji] <- diversity(vec, index = "shannon")
    dyxD$Pielou.E[dyxD$jijie == ji] <- diversity(vec, index = "shannon")/log(specnumber(vec))
    dyxD$DM[dyxD$jijie == ji] <- Margalef.DM(vec)
  }
}else{
  df <- myd %>%
    group_by(中文名) %>%
    summarise(nums = sum(数量),
              .groups = "drop")
  
  df <- df%>%
    mutate(优势度 = sprintf("%.2f%%", nums/sum(nums)*100))
  
  dyxD <- data.frame(matrix(0, 1, 5))
  dyxD[1, 1] <- "niaolei"
  colnames(dyxD) <- c('多样性指数', 'Simpson.D', 'Shannon.H','Pielou.E','DM')
  
  dyxD$Simpson.D <- diversity(df$nums, index = "simpson")
  dyxD$Shannon.H <- diversity(df$nums, index = "shannon")
  dyxD$Pielou.E <- diversity(df$nums, index = "shannon")/log(specnumber(vec))
  dyxD$DM <- Margalef.DM(df$nums)

}


write.csv(dyxD, file.path(work_dir, 'dyxb.csv'),fileEncoding = 'GB18030',row.names=F)

summary <- paste0(
  "【分析完成】\n",
  "输出文件：\n",
  "1. dyxb：", file.path(work_dir, "dyxb.csv"), "\n",
  "2. 多样性图：", file.path(work_dir, "diversity_facet_plot.pptx"), "\n"
)
cat(summary)