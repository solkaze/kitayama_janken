import threading

print("ch")

def increment(shared_value, lock):
    for _ in range(100):
        with lock:
            shared_value[0] += 1

if __name__ == '__main__':
    shared_value = [0]  # リストの要素を共有することで可変な値を使う
    lock = threading.Lock()
    threads = []

    for _ in range(10):
        t = threading.Thread(target=increment, args=(shared_value, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f'Final value: {shared_value[0]}')
