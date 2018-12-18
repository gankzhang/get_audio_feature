import os
def get_data_dir():
    data_dir = './data/纯音乐的殿堂_900_918'
    return data_dir
def get_stop_dir():
    stop_dir = './data/stopwords.txt'
    return stop_dir
def get_audio_dir():
    return './data'
#    return './data/适合咖啡厅、阅读的安静纯音乐'  # /A Little Story'
#    return './audio_data'  # /A Little Story'
def get_save_dir():
#    return os.path.join('./data/model/rnn/','model.ckpt')
    return './data/model/'
def get_embed_dim():
    return 256
def if_segment():
    return False
def wavenet_param_dir():
    return './data/wavenet_params.json'