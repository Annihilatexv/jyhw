import time
import threading
import argparse
from collections import deque
from openvinoModel import OpenVINOModel

def test_latency(model: OpenVINOModel, num_requests: int):
    """
    测试单次推理延迟。
    """
    print(f"\n--- Running Latency Test ({num_requests} requests) ---")
    dummy_input = model.create_dummy_input()
    latencies = []

    # 预热一次，确保模型JIT编译完成
    print("Warming up...")
    model.infer(dummy_input)

    print("Starting benchmark...")
    for i in range(num_requests):
        start_time = time.perf_counter()
        model.infer(dummy_input)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        print(f"Request {i+1}/{num_requests}: {latency_ms:.4f} ms")

    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print("-" * 30)
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"P99 Latency:     {p99_latency:.4f} ms")
    print("-" * 30)

def test_throughput(model: OpenVINOModel, duration_sec: int):
    """
    在固定时间内测试最大吞吐量。
    """
    print(f"\n--- Running Throughput Test ({duration_sec} seconds) ---")
    dummy_input = model.create_dummy_input()
    
    # 预热
    print("Warming up...")
    model.infer(dummy_input)

    print("Starting benchmark...")
    request_count = 0
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration_sec:
        model.infer(dummy_input)
        request_count += 1
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    throughput = request_count / elapsed_time

    print("-" * 30)
    print(f"Total requests: {request_count}")
    print(f"Elapsed time:   {elapsed_time:.4f} s")
    print(f"Throughput (QPS): {throughput:.4f}")
    print("-" * 30)

def test_concurrency(model_path: str, device: str, num_threads: int, duration_sec: int):
    """
    测试多线程并发下的吞吐量。
    每个线程有自己的模型实例和推理请求。
    """
    print(f"\n--- Running Concurrency Test ({num_threads} threads, {duration_sec} seconds) ---")
    
    total_requests = 0
    lock = threading.Lock()

    def worker():
        nonlocal total_requests
        # 每个线程创建自己的模型实例
        local_model = OpenVINOModel(model_path, device)
        local_model.compile()
        dummy_input = local_model.create_dummy_input()

        # 预热
        local_model.infer(dummy_input)

        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration_sec:
            local_model.infer(dummy_input)
            with lock:
                total_requests += 1

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    print(f"Started {num_threads} worker threads...")

    start_time_global = time.perf_counter()
    for t in threads:
        t.join()
    end_time_global = time.perf_counter()
    
    elapsed_time = end_time_global - start_time_global
    throughput = total_requests / elapsed_time

    print("-" * 30)
    print(f"Total requests from all threads: {total_requests}")
    print(f"Total elapsed time: {elapsed_time:.4f} s")
    print(f"Aggregated Throughput (QPS): {throughput:.4f}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="OpenVINO Workload Generator")
    parser.add_argument("-m", "--model", required=True, help="Path to ONNX or IR model.")
    parser.add_argument("-d", "--device", default="CPU", help="Device to run on (e.g., CPU, GPU).")
    parser.add_argument("-t", "--test_type", required=True, choices=['latency', 'throughput', 'concurrency'],
                        help="Type of test to run.")
    parser.add_argument("--num_requests", type=int, default=100, help="Number of requests for latency test.")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds for throughput/concurrency tests.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for concurrency test.")
    
    args = parser.parse_args()

    # For latency and throughput, we use a single model instance
    if args.test_type in ['latency', 'throughput']:
        # 在这些模式下，我们通常不关心算子性能，所以关闭profiling以获得最佳性能
        model = OpenVINOModel(args.model, args.device)
        model.compile(enable_profiling=False)
        
        if args.test_type == 'latency':
            test_latency(model, args.num_requests)
        elif args.test_type == 'throughput':
            test_throughput(model, args.duration)
            
    elif args.test_type == 'concurrency':
        test_concurrency(args.model, args.device, args.num_threads, args.duration)

if __name__ == "__main__":
    main()