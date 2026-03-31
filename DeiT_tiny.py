import torch
import timm
import torch.utils.benchmark as benchmark

def build_model(pretrained=False):
    torch.manual_seed(0)
    model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained)
    model.eval()
    return model

def build_input(b=1, c=3, h=224, w=224):
    torch.manual_seed(0)
    x = torch.randn(b, c, h, w)
    return x

def run_benchmark(model, x, repeat=20, threads=1):
    with torch.inference_mode():
        for _ in range(10):
            model(x)

        timer = benchmark.Timer(
            stmt="model(x)",
            globals={"model": model, "x": x},
            num_threads=threads,
            label="DeiT-tiny"
        )
        return timer.timeit(repeat)

def main():
    model = build_model()
    x = build_input()

    result = run_benchmark(model, x)
    print(result)

    y = model(x)
    print("this is y[0, :10]: ", y[0, :10])
    
    return 0


if __name__ == "__main__":
    main()

