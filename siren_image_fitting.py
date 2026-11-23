"""
G3 任务实现

目标：
    用 MLP 拟合一张图片：输入像素坐标 (x, y)，输出该点的 RGB 三个值。
    训练完成后，网络权重就是这张图的“压缩表示”。

核心（和普通 MLP 只差两点）：
    1. 激活函数用 sin(30 * x)，不是 ReLU
    2. 权值初始化要配套：中间层除以 w0，第一层单独一套

"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from PIL import Image
from torch import optim
from torchvision.transforms import Compose, Normalize, ToTensor


# 微软雅黑既有中文也有 U+2212，可以同时满足标题和公式刻度。
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TARGET_IMAGE = './data/alba-photo-027.png'

# 通用常量
SIZE = 256      # 图片尺寸
STEPS = 10000   # 训练步数
LR_RATE = 1e-4  # 学习率
ACT = 'sine'    # 激活函数

# 图片的加载和处理, size缩放到指定尺寸
def load_target_image(size=512):
    img = Image.open(TARGET_IMAGE).convert('RGB')
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return img

# 生成展平的像素坐标网络
def get_mgrid(sidelen: int):

    grid = np.mgrid[:sidelen, :sidelen]

    coords = np.stack(grid, axis=-1).astype(np.float32)

    # 把 [0, sidelen-1] 压缩到 [0, 1] 再压缩到 [-1, 1]
    coords = coords/(sidelen-1) # 现在的范围是[0,1]
    coords = coords * 2.0 - 1.0 # 现在的范围是[-1,1]

    # 转换为 PyTorch 张量, 形状为 (sidelen^2, 2)
    tensor1 = tc.from_numpy(coords).view(-1, 2)

    return tensor1

# 然后就可以创建数据集了
def create_dataset(size=512):
    """
    构造的是 (坐标, 像素值) 数据集。

    """
    # 先读图
    img = load_target_image(size)

    # 转张量加归一化
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor = transform(img)
    channels = tensor.shape[0]

    # 将HWC展平得到 (N, 3)
    target = tensor.permute(1, 2, 0).view(-1, channels)

    # 生成坐标网络
    coords = get_mgrid(size)

    return coords, target, img

# 构建神经网络
class Sine(nn.Module):
    """正弦激活：sin(w0 * x)。w0 越大，能表达越高频的细节，但是也会纳入更多噪声。"""

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return tc.sin(self.w0 * x)

def sine_init(module, w0=30.0):
    """中间层初始化：U(-sqrt(6/n)/w0, sqrt(6/n)/w0)。"""
    if isinstance(module, nn.Linear):
        with tc.no_grad():
            n = module.weight.size(-1)
            bound = math.sqrt(6.0 / n) / w0
            module.weight.uniform_(-bound, bound)

def first_sin_init(module):
    """输入层初始化：U(-1/n, 1/n)。"""
    if isinstance(module, nn.Linear):
        with tc.no_grad():
            n = module.weight.size(-1)
            module.weight.uniform_(-1/n, 1/n)

class SirenNet(nn.Module):
    """SIREN 网络。

    in_features: 输入特征数，即坐标维度
    out_features: 输出特征数，即像素值维度
    hidden_features: 隐藏层特征数，即每层的节点数
    hidden_layers: 隐藏层层数，即线性层个数
    act: 激活函数，可选 'sine' 或 'relu'
    w0: 正弦激活的参数
    """
    def __init__(self, in_features=2, out_features=3, hidden_features=256,
                 hidden_layers=3, act='sine', w0=30.0):
        super().__init__()

        # 1. 按照“线性层 + 激活函数”拼凑网络
        def make_act():
            return Sine(w0) if act == 'sine' else nn.ReLU(inplace=True)

        layers = [nn.Linear(in_features, hidden_features), make_act()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_features, hidden_features), make_act()]

        # 2. 封装输出层
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)   # 应用结构体

        # 3. 权值初始化  (未初始化尝试注释)
        # if act == 'sine':
        #     self.net.apply(lambda m: sine_init(m, w0))
        #     first_sin_init(self.net[0])

    def forward(self, coords):
        return self.net(coords)


# 模型训练
def train_model(size:int, steps:int, lr:float|int, act:str):
    """
    训练 SIREN 拟合彩色图像。
    :param size: 图像边长
    :param steps: 训练步数
    :param lr: 学习率
    :param act: 激活函数
    """

    # 1. 准备数据
    coords, target, img = create_dataset(size)

    # 2. 选择设备
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    coords = coords.to(device)
    target = target.to(device)
    channels = target.shape[-1]
    print(f'设备：{device}，数据集大小：{coords.shape} -> {target.shape}，通道数：{channels}，激活函数：{act}')

    # 3. 创建模型、损失函数、优化器
    model = SirenNet(out_features=channels, act=act).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 训练循环
    loss_list = []
    for step in range(steps+1):
        # 每2000步保存一次图像
        if step % 2000 == 0:
            show_result(model, coords, target, size, step)
            model.train()

        pred = model(coords)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        if step % 100 == 0:  # 每100步打印一次损失
            print(f'step: {step:4d}, loss: {loss.item():.7f}')

    print(f'训练完成，最终损失为：{loss_list[-1]:.7f}')
    show_result(model, coords, target, size, steps, end=True)

    # 5. 绘制损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(loss_list)
    plt.yscale('log')
    plt.title('损失曲线')
    plt.xlabel('步数')
    plt.ylabel('损失值')
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'./data/siren_data/siren未配准损失曲线_{act}.png', dpi=130)
    plt.show()

    # 6. 保存模型
    # tc.save(model.state_dict(), f'./model/siren彩图模型.pth')
    # print(f'模型已保存到 ./model/siren彩图模型.pth')

# 模型展示
def show_result(model, coords, target, size, step, end=False):

    # 1. 整图推理（no_grad 省显存）
    model.eval()
    with tc.no_grad():
        pred = model(coords).view(size, size, -1)

    # 2. 反归一化
    pred = ((pred + 1.0) / 2.0).clamp(0,1).cpu().numpy()
    gt = ((target.view(size, size, -1) + 1.0) / 2.0).clamp(0,1).cpu().numpy()

    # 3. 算 PSNR 值，单位 dB
    mse = np.mean((pred - gt)**2)
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')
    print(f'step: {step:4d}, psnr: {psnr:.2f} dB')

    # 4. 保存过程图片
    if not end:
        plt.imshow(pred)
        plt.savefig(f'./data/siren_data/siren未配准过程图片_{step}.png', dpi=130)
        plt.show()
        plt.close()

    # 5. 如果是最后存图
    if end:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes, [gt, pred, np.abs(gt - pred)],
                                  ['原图', 'siren未配准 重建', '误差']):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('./data/siren_data/siren未配准结果图片.png', dpi=130)
        plt.show()

if __name__ == '__main__':

    train_model(size=SIZE,steps=STEPS,lr=LR_RATE,act=ACT)
    # 把各训练步数的结果图横向拼成一张，方便看“收敛过程”
    # 关键点：文件名由 step 拼出来，读图、标题、子图数量全都跟着 steps 走，
    # 以后想加减对比的步数（比如加上 200/400），只改下面这一行就够了。

    steps = [2000 * i for i in range(6)]
    images = [plt.imread(f'./data/siren_data/siren未配准过程图片_{step}.png') for step in steps]

    fig, axes = plt.subplots(1, len(steps), figsize=(16, 4))
    # 只有一个子图时 subplots 返回的是单个 Axes 而不是数组，包一层保证可迭代
    for i, (ax, img, step) in enumerate(zip(np.atleast_1d(axes), images, steps), start=1):
        ax.imshow(img)
        ax.set_title(f'第 {i} 张（{step} 步）')   # 标题带上步数，比“第几张”信息量更大
        ax.axis('off')
    fig.tight_layout()
    fig.savefig('./data/siren_data/siren未配准步骤进化图.png', dpi=130)
    plt.show()