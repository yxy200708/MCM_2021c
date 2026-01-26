# 1. 加载必要库
library(jpeg)
library(data.table)
library(parallel)

# --- 配置区 ---
base_dir <- "D:/wzm/python_math_model"
img_path <- file.path(base_dir, "2021MCM_ProblemC_Files")
output_csv <- file.path(base_dir, "Full_Image_Features_SVD.csv")

# 获取所有图片路径
all_imgs <- list.files(img_path, pattern = "\\.(jpg|jpeg|png)$", 
                       full.names = TRUE, ignore.case = TRUE)
total_count <- length(all_imgs)
message(paste(">>> 检测到图片总数:", total_count))

# --- 核心处理函数 (增强容错版) ---
process_svd_full <- function(file) {
  # 增加错误捕捉，防止单张损坏图报错中断整个并行任务
  s_values <- tryCatch({
    img <- jpeg::readJPEG(file)
    # 灰度化
    gray <- if(length(dim(img)) == 3) {
      0.299 * img[,,1] + 0.587 * img[,,2] + 0.114 * img[,,3]
    } else { img }
    # 提取前10个奇异值
    svd(gray)$d[1:10]
  }, error = function(e) {
    return(rep(NA, 10)) # 损坏图片返回 NA
  })
  return(s_values)
}

# --- 并行计算启动 ---
message(">>> 正在启动 14 核并行计算，请稍候...")
# 建议保留 4 个核心供系统运行，使用 14 个核心
cl <- makeCluster(14) 
clusterExport(cl, "process_svd_full")
clusterEvalQ(cl, library(jpeg))

start_time <- Sys.time()
results <- parLapply(cl, all_imgs, process_svd_full)
stopCluster(cl)
end_time <- Sys.time()

# --- 整合与清洗 ---
message(">>> 计算完成，正在整合并保存数据...")
dt_full <- as.data.table(do.call(rbind, results))
setnames(dt_full, paste0("V", 1:10))
dt_full[, FileName := basename(all_imgs)]

# 统计成功率
success_count <- nrow(dt_full[!is.na(V1)])
message(paste(">>> 处理完毕！耗时:", round(difftime(end_time, start_time, units="mins"), 2), "分钟"))
message(paste(">>> 成功:", success_count, "张 | 失败:", total_count - success_count, "张"))

# --- 导出纯特征表 ---
fwrite(dt_full, output_csv)
message(paste(">>> 全量特征表已存至:", output_csv))