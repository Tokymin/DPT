import torch
import torch.nn as nn
import torch.nn.functional as F
# from diffusers.models.embeddings import TimestepEmbedding
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import diffusers.models


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = sample.unsqueeze(-1)  # 确保 sample 形状为 [batch_size, 1]
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample.squeeze(-1)  # 确保返回形状为 [batch_size, time_embed_dim]


class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(ZeroConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, (1, 1), bias)
        nn.init.zeros_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        self.groups = 1

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ControlNet(nn.Module):
    def __init__(self, pretrained_model, in_channels, condition_channels):
        super(ControlNet, self).__init__()
        self.pretrained_model = pretrained_model
        # 定义ControlNet的层
        self.trainable_conv_in = ZeroConv2d(in_channels=in_channels + condition_channels, out_channels=320,
                                            kernel_size=3, padding=1)
        self.trainable_conv_out = ZeroConv2d(1280, 4, kernel_size=3, padding=1)  # 确保输入通道数为1280

        # # 定义中间层，根据预训练模型的结构
        self.trainable_time_proj = None
        self.trainable_time_embedding = TimestepEmbedding(in_channels=1, time_embed_dim=1280)
        # 初始化为预训练模型的参数
        self.trainable_down_blocks = pretrained_model.down_blocks
        self.trainable_mid_block = pretrained_model.mid_block
        self.trainable_up_blocks = pretrained_model.up_blocks
        # self._initialize_weights_from_pretrained()
        # 冻结预训练模型的参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x, condition):

        print(f"forward start,x shape is {x.shape}")
        x = torch.cat([x, condition], dim=1)
        x = self.trainable_conv_in(x)  # Pass through trainable conv_in
        print(f"after trainable_conv_in,x shape is {x.shape}")
        # Generate temb
        batch_size = x.size(0)
        time_steps = torch.arange(0, batch_size, device=x.device).float()  # 生成时间步长
        print(f"time_steps shape: {time_steps.shape}")
        temb = self.trainable_time_embedding(time_steps)
        print(f"Generated temb, shape is {temb.shape}")
        # Initialize encoder_hidden_states (这里可能需要根据实际情况进行调整)
        # Initialize encoder_hidden_states and adjust shape
        encoder_hidden_states = x  # .view(batch_size, -1)  # 展平张量
        print(f"encoder_hidden_states shape after view: {encoder_hidden_states.shape}")

        print(self.trainable_down_blocks)

        exit(0)

        for block in self.trainable_down_blocks:  # Pass through down blocks
            if block is not None:
                try:
                    x = block(x, temb, encoder_hidden_states=encoder_hidden_states)
                    print(f"after block {type(block).__name__}, x shape is {x.shape}")
                except Exception as e:
                    print(f"Error in block {type(block).__name__}: {e}")
                    raise

        print(f"after trainable_down_blocks,x shape is {x.shape}")
        if self.trainable_mid_block is not None:  # Pass through the middle block
            x = self.trainable_mid_block(x)
        print(f"after trainable_mid_block,x shape is {x.shape}")
        for block in self.trainable_up_blocks:  # Pass through the up blocks
            if block is not None:
                x = block(x)
        x = self.trainable_conv_out(x)  # Pass through trainable conv_out
        print(f"forward end,x shape is {x.shape}")
        return x


class StableDiffusionWithControlNet(nn.Module):
    def __init__(self, pipeline, controlnet):
        super(StableDiffusionWithControlNet, self).__init__()
        self.pipeline = pipeline
        self.controlnet = controlnet

    def forward(self, x, condition):
        condition_output = self.controlnet(x, condition)
        generated_image = self.pipeline(condition_output)["sample"]
        return generated_image


class CustomDataset(Dataset):
    def __init__(self, image_paths, condition_paths, transform=None):
        self.image_paths = image_paths
        self.condition_paths = condition_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        condition = Image.open(self.condition_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
            condition = self.transform(condition)
        return image, condition


if __name__ == '__main__':
    from diffusers import StableDiffusionPipeline

    # ===========================检查CUDA是否可用==========================
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for computation.")
    else:
        print("CUDA is not available. Using CPU for computation.")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    model_id = r"J:\Projects\stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda")
    pretrained_model = pipeline.unet
    controlnet = ControlNet(pretrained_model, in_channels=1, condition_channels=3).to("cuda")
    combined_model = StableDiffusionWithControlNet(pipeline, controlnet)

    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # assert torch.equal(controlnet.trainable_conv_in.weight.data, pretrained_model.conv_in.weight.data)
    # assert torch.equal(controlnet.trainable_conv_in.bias.data, pretrained_model.conv_in.bias.data)

    for param in controlnet.pretrained_model.parameters():
        assert param.requires_grad == False
    print("ControlNet initialized with pretrained model parameters and ready for training.")
    # print(controlnet)
    # x = torch.randn(9, 6, 320, 320)  # 假设的输入数据
    # # conv = ZeroConv2d(6, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # conv = ZeroConv2d(6, 1280, kernel_size=3, padding=1)
    # y = conv(x)
    # print(y.shape)  # 检查输出形状
    # exit(0)
    depth_gpt = torch.randn(9, 1, 320, 320).to("cuda")
    condition = torch.randn(9, 3, 320, 320).to("cuda")
    outputs = controlnet(depth_gpt, condition=condition)
