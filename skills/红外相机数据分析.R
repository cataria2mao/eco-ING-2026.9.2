library(dplyr)
library(readxl)
library(tidyr)
library(ggplot2)
library(ggalt)
library(export)
library(scales)
library(overlap)
library(activity)
library(lubridate)
library(terra)
library(hms)

setwd('F:/玛尔挡/报告撰写')

# 读取数据
myd <- read_excel('MEDwork.xlsx', sheet = "综合")


# 时间数据转换
myd$Time <- as_hms(format(myd$拍摄时间, "%T"))
myd$hour <- hour(myd$Time)
myd$decimal_time <- hour(myd$Time)+minute(myd$Time)/60+second(myd$Time)/3600
myd$decimal_time <- myd$decimal_time/24
myd$Time_seconds <- as.numeric(myd$Time, "seconds")


species_mashe <- myd %>% 
  filter(物种名称 == "马麝", 
         独立探测首张 == 1)

species_data2 <- myd %>% 
  filter(物种名称 == "马鹿", 
         独立探测首张 == 1)

species1<-species_mashe$Time_seconds*2*pi/(24*60*60)

species2<-species_data2$Time_seconds*2*pi/(24*60*60)


# 绘制单个物种密度图
densityPlot(species1, rug = TRUE, main = "马麝日活动节律", 
            xlab = "时间", ylab = "活动强度")
graph2ppt(file="马麝日活动节律.pptx")


densityPlot(species2, rug = TRUE, main = "马鹿日活动节律", 
            xlab = "时间", ylab = "活动强度")
graph2ppt(file="马鹿日活动节律.pptx")


species_mashe_summer <- myd %>% 
  filter(物种名称 == "马麝", 
         独立探测首张 == 1,
         季节 == "夏季")


species_mashe_autumn <- myd %>% 
  filter(物种名称 == "马麝", 
         独立探测首张 == 1,
         季节 == "秋季")


# 计算重叠系数
spec1spec2est <- overlapEst(species_mashe_summer, species_mashe_autumn, type = "Dhat4")

print(spec1spec2est)


# 95% 置信区间
boot <- bootstrap(species_mashe_summer, species_mashe_autumn, nb = 1000, type = "Dhat4")
CI <- quantile(boot, probs = c(0.025, 0.975), na.rm = TRUE)

print(CI)


#显著性检验
fit1 <- fitact(species_mashe_summer)
fit2 <- fitact(species_mashe_autumn)

# 执行置换检验（设置较大的重抽样次数，如 9999）
test_result <- compareCkern(fit1, fit2, reps = 9999)

print(test_result)


# 绘制重叠图
overlapPlot(species_mashe_summer, species_mashe_autumn,
            main = paste("马麝夏秋季日活动节律重叠率: Δ =", round(spec1spec2est, 3)),
            xlab = "时间", ylab = "活动强度")
legend("topright", legend = c("夏季", "秋季"), 
       col = c("blue", "black"), lty = 1, lwd = 2)

graph2ppt(file="马麝夏秋季日活动节律重叠.pptx")

# 修改输出信息
cat("马麝夏季独立探测数:", length(species_mashe_summer), "\n")
cat("马麝秋季独立探测数:", length(species_mashe_autumn), "\n")
cat("重叠系数 Δ:", spec1spec2est, "\n")


dev.off()


#夜间相对丰富度
INRAdata <- myd %>%
  filter(
    物种名称 %in% c("马麝", "马鹿", "岩羊"),
    独立探测首张 == 1
  ) %>%
  group_by(物种名称) %>%
  summarise(
    夜间计数 = sum(hour %in% c(20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8), na.rm = TRUE),
    总计数 = sum(独立探测首张)
  )
INRAdata$INRA <- round(INRAdata$夜间计数/INRAdata$总计数, 3)


p4 <- ggplot(INRAdata, aes(x = 物种名称, y = INRA))+
  geom_bar(stat="identity",width=0.3,
           fill = "steelblue",
           colour = "steelblue")+
  geom_text(aes(label = INRA), vjust = -0.5) +
  scale_y_continuous(limits = c(0,0.65),expand = c(0,0))+
  theme_bw()+
  theme(axis.text.x = element_text(size = 12), 
        axis.text.y = element_text(size = 12))
 
p4

graph2ppt(file="INRA.pptx")


#区域相对丰富度
#导入数据
RAIdf <- myd
#生境区域工作天数表
RAIdf_day <- read_excel('MEDwork.xlsx', sheet = 'Sheet3')

#NA替换为0
RAIdf$独立探测首张[is.na(RAIdf$独立探测首张)] <- 0


species_list <- c("马麝", "马鹿", "岩羊", "中华鬣羚", "野猪", "狍")

#不同生境工作天数
RAIdf_day_sj <- RAIdf_day %>%
  group_by(生境) %>%
  summarise(有效工作日 = sum(工作天数))

RAIdf_species <- RAIdf %>%
  filter(物种名称 %in% species_list) %>%
  group_by(生境, 物种名称) %>%
  summarise(独立探测首张 = sum(独立探测首张), .groups = "drop") %>%
  tidyr::pivot_wider(
    names_from = 物种名称,
    values_from = 独立探测首张,
    values_fill = 0
  )

#合并数据并计算相对多度指数（RAI）
RAIdf_ytl_day <- RAIdf_day_sj %>%
  left_join(RAIdf_species, by = "生境") %>%
  # 将NA替换为0（确保所有物种列都存在）
  mutate(across(all_of(species_list), ~ ifelse(is.na(.), 0, .))) %>%
  # 计算RAI（每100个有效工作日的独立探测数）
  mutate(
    across(
      all_of(species_list),
      ~ . / 有效工作日 * 100,
      .names = "{.col}per"
    )
  )

#各个区域工作天数
RAIdf_day_qy <- RAIdf_day %>%
  group_by(区域) %>%
  summarise(有效工作日 = sum(工作天数))



#不同区域单个相机的有蹄类独立有效照片数
RAIdf_js <- RAIdf[RAIdf$物种名称 %in% c("马麝", "岩羊", "马鹿",
                                    "野猪", "中华鬣羚", "狍"), ] %>%
  group_by(区域, 编号) %>%
  summarise(独立有效照片数 = sum(独立探测首张),
            .groups = "drop")

RAIdf_js <- merge(RAIdf_js, RAIdf_day, all.x = T)

#不同区域单个相机的有蹄类拍摄率
RAIdf_js$单个相机拍摄率 <-RAIdf_js$独立有效照片数/RAIdf_js$工作天数

write.csv(RAIdf_js, 'raidf_js2.csv', fileEncoding = 'GB18030', row.names = FALSE)

RAIdf_js <- read_excel('raidf_js2.xlsx', sheet = 'raidf_js2')

t.test(RAIdf_js[RAIdf_js$生境 == "河流岸边", 12])

t.test(RAIdf_js[RAIdf_js$区域 == "拉则拉", 12])

cc <- c(0, 0.333)
t.test(cc)


RAIdf_sj <- RAIdf %>%
  group_by(生境) %>%
  summarise(独立有效照片数 = sum(独立探测首张),
            .groups = "drop")

RAIdf_sj_s <- RAIdf %>%
  group_by(生境, 物种名称) %>%
  summarise(独立有效照片数 = sum(独立探测首张),
            .groups = "drop")%>%
  pivot_wider(names_from = 物种名称,
              values_from = 独立有效照片数,
              values_fill = 0
            )

RAIdf_sj_s <- merge(RAIdf_sj, RAIdf_sj_s, all = T)

RAIdf_sw <- RAIdf %>%
  group_by(区域, 物种名称) %>%
  summarise(独立有效照片数 = sum(独立探测首张),
            .groups = "drop")

RAIdf_sw_s <- RAIdf %>%
  group_by(编号, 物种名称) %>%
  summarise(独立有效照片数 = sum(独立探测首张),
            .groups = "drop")

RAIdf_sw_s <- merge(RAIdf_day_s, RAIdf_js, all = T)


RAIdf_OUT <- merge(RAIdf_day_s, RAIdf_js, all = T)



write.csv(RAIdf_sw, 'raidf_sw.csv', fileEncoding = 'GB18030', row.names = FALSE)
write.csv(RAIdf_OUT, 'raidf_OUT.csv', fileEncoding = 'GB18030', row.names = FALSE)
write.csv(RAIdf_sw_s, 'raidf_sw_s.csv', fileEncoding = 'GB18030', row.names = FALSE)
write.csv(RAIdf_sj_s, 'raidf_sj_s.csv', fileEncoding = 'GB18030', row.names = FALSE)

ytqt <- read_excel('种群大小.xlsx', sheet = 'Sheet1')
ytqt$独立探测首张[is.na(ytqt$独立探测首张)] <- 0

msytqt <- ytqt[ytqt$物种名称 == "马麝"&ytqt$独立探测首张 == 1,]
t.test(msytqt$数量)
