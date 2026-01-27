from bing_image_downloader import downloader

def download_other_bees():
    # 1. 定义除了亚洲巨型胡蜂以外的目标
    # 这里的 key 将成为 Hornet_Dataset 下的子文件夹名
    targets = {
        "European_Hornet": "Vespa crabro wild",
        
        "Yellowjacket": "Vespula germanica yellowjacket",
        "Bumblebee": "Bombus terrestris bumblebee"
    }
    
    # 2. 爬取设置
    count_per_type = 150
    # 指定存入你已有的 Hornet_Dataset 文件夹
    save_path = "Hornet_Dataset" 

    for folder_name, keyword in targets.items():
        print(f"\n🚀 正在抓取新类别: {folder_name}...")
        
        downloader.download(
            keyword,
            limit=count_per_type,
            output_dir=save_path, 
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
            verbose=True
        )

if __name__ == "__main__":
    download_other_bees()
    print("\n✅ 所有新类别抓取完毕！请前往 Hornet_Dataset 文件夹查看。")