from bing_image_downloader import downloader
import os

def start_download(query_string, limit_count, folder_name):
    # 这里的 output_dir 就是本地保存的总目录
    # query_string 是搜索关键词
    # limit 是下载数量
    downloader.download(
        query_string, 
        limit=limit_count, 
        output_dir='Hornet_Dataset', 
        adult_filter_off=True, 
        force_replace=False, 
        timeout=60,
        verbose=True
    )

if __name__ == "__main__":
    # 1. 下载大黄蜂（实验组）
    print("正在抓取亚洲巨型大黄蜂...")
    start_download("Asian giant hornet", 150, "Real_Hornet")

    # 2. 下载普通胡蜂（对照组）
    print("正在抓取普通欧洲胡蜂...")
    start_download("European hornet vespula", 150, "Fake_Hornet")