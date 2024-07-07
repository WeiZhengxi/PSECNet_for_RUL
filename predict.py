# --------------------------------------------------------
# 论文：《A parallel multi-feature analysis model based on self-attention encoder and convolution for bearing remaining useful life prediction》
# 数据集：IEEE 2012 PHM Prognostic Challenge dataset
# Open Source Project
# Written by Yang xi ming and Wei Zheng xi
# --------------------------------------------------------

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import fftpack
from scipy.stats import kurtosis
import math

config = {
    "窗口滑动距离": 630,  # 每个数据10秒，滑动距离越大，得到的样本越多，但是可能得到无效样本影响训练
    "窗口大小": 100,  # 窗口大小设置，每个数据10秒 窗口大小/ 综合特征图下采样率 为整数
    "全局Dropout率": 0.6,  # 通过适当增大 全局Dropout率，可以减轻过拟合
    "betas":(0.1,0.999),
    "保留的频率特征范围": 40,
}
operating_conditions = [[1800, 4000], [1650, 4200], [1500, 5000]]  # [转/分钟，牛]分别代表三种工况
test_operating_conditions = {
    # 训练集每个轴承的工况
    0: operating_conditions[0],
    1: operating_conditions[0],
    2: operating_conditions[0],
    3: operating_conditions[0],
    4: operating_conditions[0],
    5: operating_conditions[1],
    6: operating_conditions[1],
    7: operating_conditions[1],
    8: operating_conditions[1],
    9: operating_conditions[1],
    10: operating_conditions[2],
}
list_testing_name = [
    'Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
    'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
    'Bearing3_3',
]
def main():
    set_random_seed(627)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 确定运行设备
    print("使用 {} 硬件运行程序.".format(device))
    testing_dataset = load_variable('testing_dataset.txt')
    test_rul_answer = np.array([5730, 2900, 1610, 1460, 7570, 7530, 1390, 3090, 1290, 580, 820])
    test_dataset = BearingDataset(testing_dataset, 'test')
    test_loader = DataLoader(test_dataset, batch_size=len(list_testing_name), shuffle=False,
                             pin_memory=True)  # batch_size必须等于样本个数，保证一次预测全部完成

    model = torch.load('best_model.pth')

    model = model.to(device)
    record_A = []
    model.eval()
    with torch.no_grad():
        for frequency, comprehensive in test_loader:
            predict = model(frequency.to(device), comprehensive.to(device))
            predict = torch.squeeze(predict, -1)
            predict = predict.cpu().numpy()
            # 计算测试损失
            predict_rul = predict

             # 计算比赛评估指标
            test_error_list = np.round(test_rul_answer - predict_rul, 0)
            Er = 100 * test_error_list / test_rul_answer  # 论文中是%Er 但公式和编程中，变量不好打%，故用Er表示论文中的%Er
            A = []  # 采用数组存储A值
            for Eri in Er:  # 依次计算A值
                A.append(calculate_Ai(Eri))
            record_A.append(A)
            Score = np.mean(A)
            print(f"11个测试集预测结果为：{predict}")
            print(
                f"测试集上11个测试轴承的预测误差为：{test_error_list.astype(int)}，\nEr（%）为：{Er.astype(int)}\n，Score = {Score:.4f}")


def calculate_Ai(Eri):
    """传入一个Eri"""
    if Eri <= 0:
        Ai = math.exp(-math.log(0.5) * (Eri / 5))
    else:
        Ai = math.exp(math.log(0.5) * (Eri / 20))
    return Ai
def time_frequency_transform(data, frequency_size):
    '''将信号经过希尔伯特变换，再经过傅里叶变换，最终输出frequency_size范围的频率特征'''
    # 由于傅里叶变换之后会出现频率共轭的现象，为了去除负频端的信号，所以先采用希尔伯特变换再进行傅里叶变换
    for index in range(data.shape[0]):  # 先进行希尔伯特变换
        data[index] = fftpack.hilbert(data[index])
    fft_window_data = np.fft.fft(data, axis=-1)  # 再进行傅里叶变换
    fft_feature = fft_window_data[:, :frequency_size]  # 滤掉高频噪声,保留0~frequency_sizeHz的频率特征
    abs_fft_feature = np.abs(fft_feature)  # 去掉虚部
    return abs_fft_feature
def statistical_feature_extraction(x):
    '''x:[水平信号，震动信号]，i_bearing用于查找工况'''
    v_signal = x[0]
    h_signal = x[1]
    # 提取 水平与竖直信号的：绝对平均，峰峰值，峭度，波形指标，裕度指标
    absolute_average_v = np.mean(np.abs(v_signal), axis=1)  # 绝对平均
    absolute_average_v = absolute_average_v.reshape(-1, 1)
    p_p_v = np.max(v_signal, axis=1) - np.min(v_signal, axis=1)  # 峰峰值
    p_p_v = p_p_v.reshape(-1, 1)
    kurtosis_v = kurtosis(v_signal, axis=1)  # 峭度
    kurtosis_v = kurtosis_v.reshape(-1, 1)
    wave_indicator_v = np.sqrt(np.mean(v_signal ** 2)) / absolute_average_v  # 波形指标
    wave_indicator_v = wave_indicator_v.reshape(-1, 1)
    clearance_v = (np.max(np.abs(v_signal), axis=1)) / ((np.mean(np.sqrt(np.abs(v_signal)), axis=1))) ** 2  # 裕度指标
    clearance_v = clearance_v.reshape(-1, 1)
    absolute_average_h = np.mean(np.abs(h_signal), axis=1)  # 绝对平均
    absolute_average_h = absolute_average_h.reshape(-1, 1)
    p_p_h = np.max(h_signal, axis=1) - np.min(h_signal, axis=1)  # 峰峰值
    p_p_h = p_p_h.reshape(-1, 1)
    kurtosis_h = kurtosis(h_signal, axis=1)  # 峭度
    kurtosis_h = kurtosis_h.reshape(-1, 1)
    wave_indicator_h = np.sqrt(np.mean(h_signal ** 2)) / absolute_average_h  # 波形指标
    wave_indicator_h = wave_indicator_h.reshape(-1, 1)
    clearance_h = (np.max(np.abs(h_signal), axis=1)) / ((np.mean(np.sqrt(np.abs(h_signal)), axis=1))) ** 2  # 裕度指标
    clearance_h = clearance_h.reshape(-1, 1)
    final_out = np.concatenate((absolute_average_v, p_p_v, kurtosis_v, wave_indicator_v, clearance_v,
                                absolute_average_h, p_p_h, kurtosis_h, wave_indicator_h, clearance_h), axis=1)
    return final_out

def extract_test_samples(bearing_data):
    '''传入所有测试轴承数据，赋值给bearing_data'''
    if True:  # 为了节约每次处理数据的时间，将所有数据处理的结果直接保存出来
        window_size = config['窗口大小']
        test_samples = []
        # 开始提取特征 -------------------------------------------------------
        for i_bearing, single_bearing_data in enumerate(bearing_data):
            single_bearing_data = np.array(single_bearing_data)
            single_bearing_v_signal = single_bearing_data[:, :, -1]  # 单个轴承竖直数据
            single_bearing_h_signal = single_bearing_data[:, :, -2]  # 单个轴承水平数据
            signal_length = len(single_bearing_v_signal)  # 计算数据长度
            # 构造绝对时间编码表
            abs_time_pos_label = [i * 10 for i in range(signal_length)]  # 每个CSV间隔10秒
            abs_time_pos_label = np.array(abs_time_pos_label).reshape(-1, 1)
            # 截取一个窗大小的数据
            window_v_data = single_bearing_v_signal[-window_size:]
            window_h_data = single_bearing_h_signal[-window_size:]
            abs_time_pos = abs_time_pos_label[-window_size:]
            # 计算统计学特征
            statistical_feature = statistical_feature_extraction([window_v_data, window_h_data], )
            # 计算固有特征
            working_condition = test_operating_conditions[i_bearing]  # 载入测试集工况条件
            working_condition = [working_condition for i in range(config["窗口大小"])]
            working_condition = np.array(working_condition)
            # 提取频率特征
            frequency_domain_feature_v = time_frequency_transform(window_v_data, config["保留的频率特征范围"])
            frequency_domain_feature_h = time_frequency_transform(window_h_data, config["保留的频率特征范围"])
            # 最终频率特征
            final_frequency_feature = np.concatenate((frequency_domain_feature_v, frequency_domain_feature_h), axis=1)
            # 最终综合特征
            final_comprehensive_feature = np.concatenate((abs_time_pos, working_condition, statistical_feature), axis=1)
            final_comprehensive_feature = final_comprehensive_feature[np.newaxis, :]  # 最前面增加通道的维度，是为了将特征构建成特征图[通道，长，宽]
            config["综合特征的尺寸"] = final_comprehensive_feature.shape[-1]
            test_samples.append([final_frequency_feature, final_comprehensive_feature])
        # 存储测试数据处理的结果
        save_variable(test_samples, r'预处理数据\testdata.txt')
    print(f"测试样本载入完成，共{len(test_samples)}个轴承作为测试样本.-----------------------------------")
    return test_samples

class BearingDataset(Dataset):
    def __init__(self, bearing_data, mode):
        self.all_data = bearing_data
        self.mode = mode
        if self.mode == 'train':
            pass
        if self.mode == 'test':
            self.sample = extract_test_samples(self.all_data)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        if self.mode == 'train':
            return (torch.tensor(self.sample[item][0], dtype=torch.float32),
                    torch.tensor(self.sample[item][1], dtype=torch.float32),
                    torch.tensor(self.label[item], dtype=torch.float32))
        elif self.mode == 'test':
            return (torch.tensor(self.sample[item][0], dtype=torch.float32),
                    torch.tensor(self.sample[item][1], dtype=torch.float32),)

def save_variable(x, filename):
    f = open(filename, 'wb')
    pickle.dump(x, f)
    f.close()
def load_variable(filename):
    f = open(filename, 'rb')
    x = pickle.load(f)
    f.close()
    return x
def set_random_seed(seed):
    """固定随机种子，保障实验可重复性"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    np.set_printoptions(precision=1)
    main()