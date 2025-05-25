from tensor import Tensor
from nn import Conv3d, Identity, relu
from nn.quantization import quantize_int8

struct TerrainGenerator:
    var repvgg_blocks: RepVGG3DBlock[]
    
    fn generate_chunk(x: Int, y: Int, z: Int) -> SparseVoxel:
        # choose activated chunk dynamicaly
        if not is_visible(x, y, z):
            return empty_voxel()
        # Rep effective reasoning
        merged_conv = self.repvgg_blocks[0].reparameterize()
        features = merged_conv(noise_map[x:y:z])
        return decode_to_voxel(features)

struct LightNet:
    var linear1: Linear
    var linear2: Linear

    fn __init__():
        self.linear1 = Linear(64, 32)
        self.linear2 = Linear(32, 16)
        quantize_int8(self)

    fn forward(light_sources: Tensor) -> Tensor:
        x = relu(self.linear1(light_sources))
        return sigmoid(self.linear2(x))
        
struct RepVGG3DBlock:
    var conv3x3: Conv3d
    var conv1x1: Conv3d
    var identity: Identity

    fn __init__(in_channels: Int, out_channels: Int):
        self.conv3x3 = Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1 = Conv3d(in_channels, out_channels, kernel_size=1)
        self.identity = Identity() if in_channels == out_channels else None

    fn forward(x: Tensor) -> Tensor:
        out = self.conv3x3(x)
        out += self.conv1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return relu(out)

    fn reparameterize() -> Conv3d:
        merged_kernel = self.conv3x3.weight + self._expand_1x1_to_3x3()
        if self.identity is not None:
            merged_kernel += self._expand_identity_to_3x3()
        merged_conv = Conv3d(..., kernel_size=3)
        merged_conv.weight = merged_kernel
        return merged_conv