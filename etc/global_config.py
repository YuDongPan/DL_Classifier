# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/31 9:31
import argparse
import yaml

config = {}


def _init():
    """
    初始化
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default="../etc/config.yaml", help='--config file')

    opt = parser.parse_args()
    config_path = opt.f
    global config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()


def get_global_conf():
    _init()
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return config
    except:
        print('读取失败\r\n')
        return {}


config = get_global_conf()