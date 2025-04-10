import torch
import time
import torch.nn as nn
import torch.optim as optim
import argparse


def get_device(preferred_device):
    """
    Returns the device based on user preference:
    - If 'cuda' is requested and available, use CUDA.
    - If 'mps' is requested and available, use MPS.
    - If 'cpu' is requested or the preferred device is not available, use CPU.
    - If 'auto' is requested, choose CUDA > MPS > CPU.
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            print("Auto mode: Using CUDA GPU.")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Auto mode: Using Apple MPS (Metal Performance Shaders).")
            return torch.device("mps")
        else:
            print("Auto mode: Using CPU.")
            return torch.device("cpu")
    elif preferred_device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA GPU.")
            return torch.device("cuda")
        else:
            print("CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
    elif preferred_device == "mps":
        if torch.backends.mps.is_available():
            print("Using Apple MPS (Metal Performance Shaders).")
            return torch.device("mps")
        else:
            print("MPS not available. Falling back to CPU.")
            return torch.device("cpu")
    elif preferred_device == "cpu":
        print("Using CPU.")
        return torch.device("cpu")
    else:
        print("Unknown device preference. Using CPU.")
        return torch.device("cpu")


def measure_time(func, device):
    """
    Measures execution time of a function.
    Uses CUDA events for precise timing on CUDA devices; otherwise, uses time.time().
    """
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        func()
        ender.record()
        torch.cuda.synchronize()
        elapsed_time_ms = starter.elapsed_time(ender)
        return elapsed_time_ms / 1000.0  # convert ms to seconds
    else:
        start = time.time()
        func()
        end = time.time()
        return end - start


def matrix_multiplication_hard_test(device):
    """
    Hard matrix multiplication: multiplies two 3000x3000 matrices.
    """

    def test():
        a = torch.randn(3000, 3000, device=device)
        b = torch.randn(3000, 3000, device=device)
        _ = torch.matmul(a, b)

    time_taken = measure_time(test, device)
    print(f"Matrix multiplication (3000x3000) time: {time_taken:.6f} seconds")


def elementwise_addition_hard_test(device):
    """
    Hard element-wise addition: adds two tensors with 100 million elements.
    """

    def test():
        a = torch.randn(100_000_000, device=device)
        b = torch.randn(100_000_000, device=device)
        _ = a + b

    time_taken = measure_time(test, device)
    print(f"Element-wise addition (100M elements) time: {time_taken:.6f} seconds")


def convolution_hard_test(device):
    """
    Hard convolution test: performs a forward pass on a large batch of high-resolution images.
    Uses a convolution layer with 3 input channels and 64 output channels.
    """

    def test():
        conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3).to(device)
        # Batch of 128 images of size 256x256
        input_tensor = torch.randn(128, 3, 256, 256, device=device)
        _ = conv(input_tensor)

    time_taken = measure_time(test, device)
    print(f"Convolution forward pass (128,3,256,256) time: {time_taken:.6f} seconds")


def reduction_hard_test(device):
    """
    Hard reduction test: sums over a tensor with 100 million elements.
    """

    def test():
        a = torch.randn(100_000_000, device=device)
        _ = torch.sum(a)

    time_taken = measure_time(test, device)
    print(f"Reduction (sum over 100M elements) time: {time_taken:.6f} seconds")


def deep_training_loop_test(device):
    """
    Hard training loop test: trains a 10-layer fully-connected network.
    Each hidden layer has 2048 units.
    Runs for 500 iterations on random data.
    """

    class DeepNet(nn.Module):
        def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=100, layers=10):
            super(DeepNet, self).__init__()
            modules = []
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(nn.ReLU())
            for _ in range(layers - 2):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*modules)

        def forward(self, x):
            return self.net(x)

    model = DeepNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Random dataset: 1024 features, target dimension 100, batch size 256
    inputs = torch.randn(256, 1024, device=device)
    targets = torch.randn(256, 100, device=device)

    def train_step():
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    def test():
        for _ in range(500):
            train_step()

    time_taken = measure_time(test, device)
    print(f"Deep training loop (500 iterations on a 10-layer net) time: {time_taken:.6f} seconds")


def transformer_block_test(device):
    """
    Hard transformer test: runs forward and backward passes through a transformer encoder.
    Uses a single encoder layer with a model dimension of 512.
    Processes a sequence of length 128 with a batch size of 32 over 100 iterations.
    """
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8).to(device)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=0.001)

    # Random data: sequence length 128, batch size 32, model dimension 512
    src = torch.randn(128, 32, 512, device=device)
    # Random target for loss computation
    target = torch.randn(128, 32, 512, device=device)

    def train_step():
        transformer_encoder.train()
        optimizer.zero_grad()
        output = transformer_encoder(src)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    def test():
        for _ in range(100):
            train_step()

    time_taken = measure_time(test, device)
    print(f"Transformer block training (100 iterations) time: {time_taken:.6f} seconds")


def main():
    parser = argparse.ArgumentParser(description="PyTorch GPU/CPU Benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for benchmarking: 'auto', 'cpu', 'cuda', or 'mps'"
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print("\nstarting heavy benchmark tests...\n")
    matrix_multiplication_hard_test(device)
    elementwise_addition_hard_test(device)
    reduction_hard_test(device)
    deep_training_loop_test(device)
    transformer_block_test(device)
    print("\nall heavy tests completed.")
    convolution_hard_test(device)

if __name__ == '__main__':
    main()
