import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import fft


def read_wav_data(filename):
    """读取一个wav文件，返回声音信号的时域谱矩阵和播放时间"""
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取帧速率
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    # wave_data = wave_data
    return wave_data, framerate


def get_mfcc_feature(wavsignal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                     nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
                     N=2):
    """
    获取音频的mfcc特征, 仅支持16khz的采样率.
    :param wavsignal: 一维的音频信号
    :param samplerate: 默认16khz采样率, 目前仅支持这个采样率
    :param winlen: 分析窗口的长度，以秒为单位。默认为0.025秒(25毫秒)
    :param winstep: 连续窗口之间以秒为单位的步骤。默认为0.01s(10毫秒)
    :param nfft: 快速傅里叶变换尺寸. 默认512.
    :param lowfreq: mel滤波器最低边带. 默认0.
    :param preemph: 预加重滤波器预处理, 默认是0.97。
    :param ceplifter: 最终倒向系数应用提升器。0代表不举重。默认是22。
    :param appendEnergy: 倒谱系数被替换为总帧能量的对数
    :return: 返回mfcc维度
    """
    feat_mfcc = mfcc(wavsignal[0], samplerate=samplerate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt,
                     nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, ceplifter=ceplifter,
                     appendEnergy=appendEnergy)
    feat_mfcc_d = delta(feat_mfcc, N)
    feat_mfcc_dd = delta(feat_mfcc_d, N)
    # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature


def get_frequency_feature(wavsignal, samplerate=16000, num_features=200, interval=160, winlen=25,
                          time_rate=1000,
                          winstep=10):
    """
    获取Frequency系数矩阵, 频谱的维度相对于mfcc来的较大, 可提取出的信息也会比较丰富.
    :param wavsignal: 一维的音频信号.
    :param samplerate: 默认16khz采样率, 目前仅支持这个采样率.
    :param num_features: 窗口长度, 也是特征数量 默认200 , 计算方式为: window_length =  [fs(16000) / 1000 * time_window(25)] // 2.
    :param interval: 滑动尺寸间隔, 为了形成连续堆叠效果
    :param winlen: 加时间窗长, 默认25, 表示25毫秒
    :param time_rate: 时率, 用于设置长度信息表示, 有标准化作用, 默认1000表示1秒
    :param winstep: 窗口的时间步长, 默认10, 表示10毫秒
    :return: 返回处理后的特征图.
    """
    assert 16000 == samplerate, "目前仅支持采样率16000HZ. "
    size = num_features * 2
    x = np.linspace(0, size - 1, size, dtype=np.int64)
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (size - 1))  # 汉明窗

    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0]) / samplerate * time_rate - winlen) // winstep  # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, num_features), dtype=np.float)  # 用于存放最终的频率特征数据

    for i in range(0, range0_end):
        p_start = i * interval
        p_end = p_start + size
        data_line = wav_arr[0, p_start:p_end]

        data_line = data_line * hamming_window  # 加窗

        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0:num_features]  # 设置为如size为400除以2的值（即200）是取一半数据，因为是对称的

    data_input = np.log(data_input + 1)
    return data_input


def wav_scale(energy):
    """语音信号能量归一化"""
    means = energy.mean()  # 均值
    var = energy.var()  # 方差
    e = (energy - means) / math.sqrt(var)  # 归一化能量
    return e


def wav_scale2(energy):
    """语音信号能量归一化"""
    maxnum = max(energy)
    e = energy / maxnum
    return e


def wav_scale3(energy):
    """语音信号能量归一化"""
    for i in range(len(energy)):
        # if i == 1:
        #	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
        energy[i] = float(energy[i]) / 100.0
    # if i == 1:
    #	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
    return energy


def wav_show(wave_data, fs):  # 显示出来声音波形
    time = np.arange(0, len(wave_data)) * (1.0 / fs)  # 计算声音的播放时间，单位为秒
    # 画声音波形
    # plt.subplot(211)
    plt.plot(time, wave_data)
    # plt.subplot(212)
    # plt.plot(time, wave_data[1], c = "g")
    plt.show()


def get_wav_list(filename):
    """读取一个wav文件列表，返回一个存储该列表的字典类型值
    ps:在数据中专门有几个文件用于存放用于训练、验证和测试的wav文件列表"""
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_filelist = {}  # 初始化字典
    list_wavmark = []  # 初始化wav列表
    for i in txt_lines:
        if (i != ''):
            txt_l = i.split(' ')
            dic_filelist[txt_l[0]] = txt_l[1]
            list_wavmark.append(txt_l[0])
    txt_obj.close()
    return dic_filelist, list_wavmark


def get_wav_symbol(filename):
    """读取指定数据集中，所有wav文件对应的语音符号
    返回一个存储符号集的字典类型值"""
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_symbol_list = {}  # 初始化字典
    list_symbolmark = []  # 初始化symbol列表
    for i in txt_lines:
        if (i != ''):
            txt_l = i.split(' ')
            dic_symbol_list[txt_l[0]] = txt_l[1:]
            list_symbolmark.append(txt_l[0])
    txt_obj.close()
    return dic_symbol_list, list_symbolmark
