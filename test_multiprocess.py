from multiprocessing import Process, Value

def increment(shared_value):
    for _ in range(100):
        with shared_value.get_lock():
            shared_value.value += 1

if __name__ == '__main__':
    shared_value = Value('i', 0)  # 整数型の共有変数
    processes = []

    for _ in range(10):
        p = Process(target=increment, args=(shared_value,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f'Final value: {shared_value.value}')
