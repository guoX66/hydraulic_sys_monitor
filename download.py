import os
import time
import zipfile
import requests


def url_download(url, filepath):
    response = requests.get(url, stream=True)
    size = 0
    chunk_size = 1024 * 1024
    try:
        if response.status_code == 200:
            start = time.perf_counter()
            with open(filepath, 'wb') as file:  # 显示进度条
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    size += len(data)
                    s = size / (1024 * 1024)
                    t = time.perf_counter() - start
                    print(
                        "\r下载大小:{:^3.1f} MB 速度:{:.2f} MB/s 用时:{:.2f} s".format(s, s / t, t),
                        end="")
                print()
        else:
            print('Connect failed！')

    except Exception as ex:
        print(ex)


def unzip(i_file, o_file):
    zip_file = zipfile.ZipFile(i_file)
    zip_file.extractall(o_file)
    zip_file.close()


if __name__ == "__main__":
    base_dir = r"data"
    os.makedirs(base_dir, exist_ok=True)
    """
        下载数据
    """
    url = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
    url_download(url, "data.zip")
    unzip('data.zip', 'data')
    os.remove('data.zip')
