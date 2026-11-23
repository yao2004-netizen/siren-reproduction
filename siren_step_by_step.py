"""
SIREN 学习脚本（逐步复现）

目标：
    用 MLP 拟合一张图片：输入像素坐标 (x, y)，输出该点的 RGB 三个值。
    训练完成后，网络权重就是这张图的“压缩表示”。
    （彩色版和灰度版的差别只有一处：输出从 1 个值变成 3 个值，坐标网格完全不变。）

核心（和普通 MLP 只差两点）：
    1. 激活函数用 sin(30 * x)，不是 ReLU
    2. 权值初始化要配套：中间层除以 w0，第一层单独一套

步骤：
    dm01  先看清目标：一张图 = 一个 (x,y) -> RGB 的函数
    dm02  造坐标网格：get_mgrid
    dm03  图像 -> 标签：归一化 + 展平成 (N, 3) + 对齐
    dm04  搭建网络：Sine / sine_init / first_layer_sine_init / SirenNet
    dm05  前向传播：打印每一层形状，确认数据怎么流动
    dm06  训练：loss 从 0.33 掉到 1e-4 量级
    dm07  可视化：loss 曲线 + 原图/重建/误差

运行：
    直接运行本文件，依次执行 dm01、dm02、dm03、dm05，再训练并出图。
    想单独试某一步，把 main 里其他调用注释掉即可。
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from PIL import Image
from torch import optim
from torchvision.transforms import Compose, Normalize, ToTensor

# 中文字体：不能用 SimHei/宋体/楷体，它们缺 U+2212 减号字形，
# 而 log 坐标的 10^-4 刻度是用 mathtext 画的，mathtext 不做逐字回退，
# 会把减号画成方框并刷一堆 “does not have a glyph for '\u2212'” 警告。
# 微软雅黑既有中文也有 U+2212，可以同时满足标题和公式刻度。
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 拟合目标：想换一张图，只改这一行（换成任意本地图片路径即可）
# 前提：图片得是正方形，因为 get_mgrid 和 reshape 都按 size x size 算。
# 非正方形的话，要把 size 拆成 (h, w) 两个参数，涉及 get_mgrid / show_result。
TARGET_IMAGE = './data/alba-photo-027.png'


# ============================================================
# 第一步：先搞清楚“要拟合的东西”长什么样
# ============================================================
def load_target_image(size):
    """读目标图并缩放到 size x size，统一成 RGB 三通道。

    convert('RGB') 不是多余的：如果原图是 RGBA（带透明通道）、L（灰度）
    或 P（调色板），ToTensor 出来的通道数就不一定是 3，
    会和模型输出层的 out_features 对不上，报 shape 错误。
    """
    return Image.open(TARGET_IMAGE).convert('RGB').resize(
        (size, size), Image.Resampling.BICUBIC)


def dm01():
    """一张图，本质就是一个函数 f(x, y) = 该点的像素值。"""

    # 1. 读取目标图片（彩色）
    img = load_target_image(256)
    img.show()
    print('图像尺寸:', img.size, '模式:', img.mode)

    # 2. 转成张量，并归一化到 [-1, 1]
    transform = Compose([ToTensor(), Normalize([0.5] * 3, [0.5] * 3)])
    tensor = transform(img)
    print('图像张量形状:', tuple(tensor.shape))       # (3, 256, 256) = (C, H, W)
    print('数值范围:', round(tensor.min().item(), 3), '~', round(tensor.max().item(), 3))

    # 3. 关键认知：这张图等价于 256*256 = 65536 条样本
    #    每条样本 = (x坐标, y坐标) -> 3个颜色值
    print('可用的训练样本数:', 256 * 256)

# ============================================================
# 第二步：造坐标网格
# ============================================================
def get_mgrid(sidelen):
    """
    生成展平的像素坐标网格。

    参数:
        sidelen: 边长，生成 sidelen x sidelen 个坐标
    返回:
        形状 (sidelen*sidelen, 2) 的张量，取值范围 [-1, 1]
    """
    # 1. mgrid 的第0层是行号(y)，第1层是列号(x)，形状 (2, H, W)
    grid = np.mgrid[:sidelen, :sidelen]

    # 2. 把两层的值叠到最后一维 -> (H, W, 2)
    coords = np.stack(grid, axis=-1).astype(np.float32)

    # 3. 把 [0, sidelen-1] 压缩到 [0, 1]，再映射到 [-1, 1]
    coords = coords / (sidelen - 1)
    coords = coords * 2.0 - 1.0

    # 4. 展平成 (N, 2)，每个点一个 (y, x)
    return tc.from_numpy(coords).view(-1, 2)


def dm02():
    """用小尺寸 4x4 把坐标网格打印出来，看得清楚。"""

    coords = get_mgrid(4)
    print('坐标网格形状:', tuple(coords.shape))   # (16, 2)
    print(coords)
    print('取值范围:', coords.min().item(), '~', coords.max().item())

    # 注意：第0列是行号(y)，第1列是列号(x)，别当成 (x, y)
    print('左上角:', coords[0].tolist(), ' 右下角:', coords[-1].tolist())


# ============================================================
# 第三步：图像 -> 标签，并保证与坐标一一对应
# ============================================================
def create_dataset(size=128):
    """
    构造 (坐标, 像素值) 数据集。

    参数:
        size: 图像边长
    返回:
        coords: (N, 2)，target: (N, 3)，img: 原始 PIL 图像
    """
    # 1. 读图并缩放
    img = load_target_image(size)

    # 2. 转张量 + 归一化到 [-1, 1]（和坐标的尺度保持一致）
    #    三个通道用同一组 mean/std，所以写 [0.5]*3 也行 [0.5] 广播也行
    transform = Compose([ToTensor(), Normalize([0.5] * 3, [0.5] * 3)])
    tensor = transform(img)                       # (3, H, W) = (C, H, W)
    channels = tensor.shape[0]

    # 3. HWC -> 展平，得到 (N, 3)
    #    permute(1,2,0) 把 CHW 换成 HWC，再按行优先展平：
    #    第 i 行 = 第 i 个像素的 (R, G, B)，和下面 coords 的第 i 行一一对应
    target = tensor.permute(1, 2, 0).view(-1, channels)

    # 4. 坐标网格，顺序必须和上面一致（都是行优先），否则点对错像素
    coords = get_mgrid(size)

    return coords, target, img


def dm03():
    """验证坐标和像素是不是一一对应。"""

    coords, target, img = create_dataset(256)
    print('坐标形状:', tuple(coords.shape))
    print('标签形状:', tuple(target.shape))       # (65536, 3)
    assert coords.shape[0] == target.shape[0], '坐标和像素数量必须相等'
    assert target.shape[1] == 3, '彩色图应该是 3 通道'

    # 第0个点 = 左上角，最后1个点 = 右下角
    print('第1个点:', coords[0].tolist(), '-> RGB', [round(v, 3) for v in target[0].tolist()])
    print('最后1个点:', coords[-1].tolist(), '-> RGB', [round(v, 3) for v in target[-1].tolist()])


# ============================================================
# 第四步：搭建 SIREN（核心就三样东西）
# ============================================================
class Sine(nn.Module):
    """正弦激活：sin(w0 * x)。w0 越大，能表达越高频的细节。"""

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return tc.sin(self.w0 * x)


def sine_init(m, w0=30.0):
    """中间层初始化：U(-sqrt(6/n)/w0, sqrt(6/n)/w0)。

    除以 w0 是为了抵消 sin 的放大效应，让每层输出保持在同一量级。
    少了这个除法，多层 sin 叠加会直接饱和，loss 卡住不动。
    """
    if isinstance(m, nn.Linear):
        with tc.no_grad():
            n = m.weight.size(-1)
            bound = math.sqrt(6.0 / n) / w0
            m.weight.uniform_(-bound, bound)


def first_layer_sine_init(m):
    """第一层单独初始化：U(-1/n, 1/n)。因为它的输入是原始坐标。"""
    if isinstance(m, nn.Linear):
        with tc.no_grad():
            n = m.weight.size(-1)
            m.weight.uniform_(-1.0 / n, 1.0 / n)


class SirenNet(nn.Module):
    """坐标 MLP：2 -> 256 -> 256 -> 256 -> 256 -> 3（R/G/B）"""

    def __init__(self, in_features=2, hidden=256, hidden_layers=3,
                 out_features=3, w0=30.0, act='sine'):
        super().__init__()

        # 1. 按“线性层 + 激活”交替拼网络
        def make_act():
            return Sine(w0) if act == 'sine' else nn.ReLU(inplace=True)

        layers = [nn.Linear(in_features, hidden), make_act()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden, hidden), make_act()]
        # 2. 输出层不加激活（回归任务输出可正可负）
        layers.append(nn.Linear(hidden, out_features))
        self.net = nn.Sequential(*layers)

        # 3. 初始化：先全网络套一遍，再单独处理第一层
        if act == 'sine':
            self.net.apply(lambda m: sine_init(m, w0))
            first_layer_sine_init(self.net[0])

    def forward(self, coords):
        return self.net(coords)


def dm04():
    """打印网络结构，数一数到底有几个 Linear。"""

    net = SirenNet(hidden=64)
    print(net)
    linear_num = sum(1 for m in net.modules() if isinstance(m, nn.Linear))
    print('Linear 层数量:', linear_num)   # 1 + 3 + 1 = 5


# ============================================================
# 第五步：前向传播，看形状怎么流
# ============================================================
def dm05():
    """每一层都打印形状，和论文里那张结构图对上。"""

    net = SirenNet(hidden=64)
    coords = get_mgrid(16)
    print('输入坐标:', tuple(coords.shape))

    x = coords
    for i, layer in enumerate(net.net):
        x = layer(x)
        print(f'第{i+1:2d}层 {layer.__class__.__name__:10s} -> {tuple(x.shape)}')

    # 输出形状 (N, 3)：每个坐标点对应 R/G/B 三个值


# ============================================================
# 第六步：训练
# ============================================================
def train_model(size=128, steps=1500, lr=1e-4, act='sine'):
    """
    训练 SIREN 拟合彩色图像。

    参数:
        size: 图像边长
        steps: 训练步数
        lr: 学习率
        act: 'sine' 用 SIREN，'relu' 用普通 MLP 做对照
    返回:
        model, coords, target, loss_list, device

    注意：彩色比灰度难拟合（细节更丰富，每个点要同时对 3 个通道），
    想要清晰的彩色重建，steps 一般要拉到 3000~5000，论文用的就是 4000。
    """
    # 1. 准备数据
    coords, target, _ = create_dataset(size)

    # 2. 选择设备
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    coords, target = coords.to(device), target.to(device)
    channels = target.shape[1]
    print(f'设备: {device}, 样本数: {coords.shape[0]}, 通道数: {channels}, 激活: {act}')

    # 3. 创建模型、损失函数、优化器
    #    输出层宽度跟着数据的通道数走，不再写死（灰度 1 / 彩色 3）
    model = SirenNet(out_features=channels, act=act).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 训练循环（这里没有 DataLoader，因为每次都喂全图）
    loss_list = []
    for step in range(steps+1):
        # 每300步保存一次图像
        if step % 800 == 0:
            show_result(model, coords, target, size, step)

        y_pred = model(coords)
        loss = loss_fn(y_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        if step % 50 == 0:
            print(f'step {step:5d}, loss: {loss.item():.6f}')
    print(f'训练结束, 最终 loss: {loss_list[-1]:.6f}')

    # 5. loss 曲线
    plt.figure(figsize=(6, 3))
    plt.plot(loss_list)
    plt.yscale('log')
    plt.title('损失曲线')
    plt.xlabel('步数')
    plt.ylabel('MSE（对数坐标）')
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.savefig('./data/siren_step_loss.png', dpi=130)
    plt.show()

    # 6. 保存模型（文件名带上通道数，不覆盖之前的灰度 checkpoint）
    tc.save(model.state_dict(), f'./model/siren_{size}_{steps}_{lr}_{act}_{channels}ch.pth')
    print(f'模型已保存: siren_{size}_{steps}_{lr}_{act}_{channels}ch.pth')
    return model, coords, target, loss_list, device


# ============================================================
# 第七步：可视化
# ============================================================
def show_result(model, coords, target, size, step):
    """把彩色重建结果画出来（并保存）。"""

    # 1. 整图推理（no_grad 省显存）
    model.eval()
    with tc.no_grad():
        # 输出是 (N, 3)，view 回 (H, W, 3)
        # 能直接 view 是因为 target 展平时就是 HWC 行优先，这个顺序可逆
        pred = model(coords).view(size, size, -1)

    # 2. 反归一化：[-1,1] -> [0,1]
    #    clamp 不能省：输出层无激活，值域可能超出 [-1,1]，
    #    超出部分会被 matplotlib 裁剪，看着像“过曝”，其实不是训练失败
    pred = ((pred + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    gt = ((target.view(size, size, -1) + 1.0) / 2.0).cpu().numpy()

    # 3. 算 PSNR，越大越接近原图（R/G/B 三通道合并算的均值）
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')
    print(f'PSNR: {psnr:.2f} dB')

    # 4. 三联图(旧)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, img, title in zip(axes, [gt, pred, np.abs(gt - pred)],
                              ['原图', 'SIREN 重建', '误差']):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig('./data/siren_step_result.png', dpi=130)
    plt.show()

    # 4. 绘制过程图片
    #    RGB 数组 (H, W, 3) 不能传 cmap='gray'：matplotlib 会直接忽略它，
    #    颜色由三个通道自己决定，留着只会误导
    plt.imshow(pred)
    # 5. 保存图像（文件名加 rgb，不覆盖旧的灰度结果图）
    plt.savefig(f'./data/siren_rgb_{step}_result.png', dpi=130)
    plt.show()
    model.train()

if __name__ == '__main__':
    # 按顺序走一遍，每一步都能独立看懂
    dm01()
    # print('-' * 50)
    # dm02()
    # print('-' * 50)
    # dm03()
    # print('-' * 50)
    # dm04()
    # print('-' * 50)
    # dm05()
    # print('-' * 50)

    # 真正训练：先把 size 调小、steps 调少，跑通再放大
    # 彩色想看得清楚，可以把 steps 拉到 4000（记得同步改下面拼接块的 steps）
    # model, coords, target, loss_list, device = train_model(
    #     size=512, steps=4000, lr=1e-4, act='sine'
    # )
    # print(model, coords, target, loss_list, device)

    # 拟合图像
    # 注意：彩色模型的输出层是 256->3，旧的灰度 checkpoint（256->1）装不进来，
    # load_state_dict 会报 shape mismatch，只能重训。
    # model = SirenNet()
    # model.load_state_dict(tc.load('./model/siren_256_1500_0.0001_sine_3ch.pth'))
    # coords, target, _ = create_dataset(256)
    # show_result(model, coords, target, size=256, step=1500)

    # 把各训练步数的结果图横向拼成一张，方便看“收敛过程”
    # 关键点：文件名由 step 拼出来，读图、标题、子图数量全都跟着 steps 走，
    # 以后想加减对比的步数（比如加上 200/400），只改下面这一行就够了。
    # steps = [0, 600, 1200, 1800, 2400, 3000, 3600]
    # images = [plt.imread(f'./data/siren_rgb_{step}_result.png') for step in steps]
    #
    # fig, axes = plt.subplots(1, len(steps), figsize=(16, 4))
    # # 只有一个子图时 subplots 返回的是单个 Axes 而不是数组，包一层保证可迭代
    # for i, (ax, img, step) in enumerate(zip(np.atleast_1d(axes), images, steps), start=1):
    #     ax.imshow(img)
    #     ax.set_title(f'第 {i} 张（{step} 步）')   # 标题带上步数，比“第几张”信息量更大
    #     ax.axis('off')
    # fig.tight_layout()
    # fig.savefig('./data/siren_rgb_step_result.png', dpi=130)
    # plt.show()

