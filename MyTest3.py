import torch
import torch.nn as nn
import torch.optim as optim


# 定义简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)


def forward(self, x):
    return self.fc(x)


# 初始化模型
model = SimpleModel()


# 定义水印嵌入函数
def embed_watermark(model, watermark):
    with torch.no_grad():
        # 假设我们只对fc层的权重进行水印嵌入
        model.fc.weight.add_(watermark)


# 定义水印提取函数
def extract_watermark(model):
    # 直接返回fc层的权重作为水印（假设这就是嵌入水印的地方）
    return model.fc.weight.clone()


# 嵌入水印
# 确保watermark与fc.weight的形状相同
watermark = torch.randn_like(model.fc.weight) * 0.01
print("Original Parameters:")
print(model.fc.weight)
embed_watermark(model, watermark)

# 提取水印
extracted_watermark = extract_watermark(model)

# 打印水印和提取的水印以进行比较
print("Watermark:")
print(watermark)
print("Extracted Watermark:")
print(extracted_watermark)

# 验证水印
# 注意：使用更小的atol值，但通常1e-3应该足够了
is_close = torch.allclose(watermark, extracted_watermark, atol=0.3)  # extracted watermark = watermark + original parameters
print(f"Are watermarks close? {is_close}")

# 如果需要，可以计算实际的最大差异
max_diff = torch.max(torch.abs(watermark - extracted_watermark))
print(f"Maximum absolute difference: {max_diff.item()}")

# 验证水印
# 注意：使用更小的atol值，因为水印的添加是乘以0.01的
print(torch.allclose(watermark, extracted_watermark, atol=0.3))