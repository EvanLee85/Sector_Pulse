from utils.scheduler import start_scheduler, shutdown_scheduler
import time

def main() -> None:
    sched = start_scheduler(auto_create_tables=True)
    try:
        # 空循环，避免进程立即退出
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_scheduler(sched)

if __name__ == "__main__":
    main()
