import time
import numpy as np
import onnxruntime as ort
import spacemit_ort


MODEL_PATH = "DeiT_tiny.onnx"
INPUT_SHAPE = (1, 3, 224, 224)

WARMUP = 10
REPEAT = 20

INTRA_OP_THREADS = 1
INTER_OP_THREADS = 1


def make_session(model_path, providers):
    so = ort.SessionOptions()
    so.intra_op_num_threads = INTRA_OP_THREADS
    so.inter_op_num_threads = INTER_OP_THREADS
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=providers
    )
    return session


def run_once(session, x):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    y = session.run([output_name], {input_name: x})[0]
    return y


def benchmark(session, x):
    # warmup
    for _ in range(WARMUP):
        run_once(session, x)

    start = time.perf_counter()
    for _ in range(REPEAT):
        run_once(session, x)
    end = time.perf_counter()

    return (end - start) / REPEAT


def run_case(name, providers, x):
    print(f"\n===== {name} =====")

    session = make_session(MODEL_PATH, providers)
    print("session providers :", session.get_providers())


    # 결과 1회 실행
    y = run_once(session, x)

    # 앞 5개 값 출력
    print("output first 5:", y.flatten()[:5])

    # 성능 측정
    avg = benchmark(session, x)
    print(f"avg sec: {avg:.6f}")

    return y


def main():
    np.random.seed(0)
    x = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    y_cpu = run_case(
        "CPU only",
        ["CPUExecutionProvider"],
        x
    )

    y_sp = run_case(
        "SpaceMIT + CPU fallback",
        ["SpaceMITExecutionProvider", "CPUExecutionProvider"],
        x
    )



if __name__ == "__main__":
    main()

