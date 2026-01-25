# 1. 加载必要库
library(jpeg)
library(data.table)

# --- 配置区 ---
img_path <- "./2021MCM_ProblemC_Files/" # 图片存放文件夹
sample_size <- 10                       # 测试样本量：先跑10张看看效果

# 获取所有图片路径
all_imgs <- list.files(img_path, pattern = "\\.(jpg|jpeg|png)$", 
                       full.names = TRUE, ignore.case = TRUE)

# 随机抽取样本进行测试
set.seed(123) # 固定随机种子，确保每次测试的图一样
test_imgs <- sample(all_imgs, min(sample_size, length(all_imgs)))
print(paste("开始测试，样本量:", length(test_imgs)))

# --- 核心处理函数 (严格遵循博文比例) ---
process_svd_single <- function(file) {
  # 1. 读取图片
  img <- tryCatch(readJPEG(file), error = function(e) NULL)
  if(is.null(img)) return(rep(NA, 10))
  
  # 2. 灰度化：0.299*R + 0.587*G + 0.114*B
  # 检查维度：彩色图为3维 (长, 宽, RGB)
  if(length(dim(img)) == 3) {
    gray <- 0.299 * img[,,1] + 0.587 * img[,,2] + 0.114 * img[,,3]
  } else {
    gray <- img # 本身就是灰度图则直接使用
  }
  
  # 3. SVD 分解并提取前 10 个奇异值
  # 奇异值代表了图片的主要特征信息
  s_values <- svd(gray)$d
  return(s_values[1:10])
}

# --- 顺序执行 (非并行) ---
message("正在处理中...")
results_list <- lapply(test_imgs, process_svd_single)

# --- 整合数据 ---
Data_test <- as.data.table(do.call(rbind, results_list))
setnames(Data_test, paste0("V", 1:10))

# 添加图片名称列，方便核对
Data_test[, FileName := basename(test_imgs)]

# --- 结果展示 ---
print("提取出的特征数据预览：")
print(Data_test)

# 打印数据结构，对照博文中的 str(Data)
str(Data_test)