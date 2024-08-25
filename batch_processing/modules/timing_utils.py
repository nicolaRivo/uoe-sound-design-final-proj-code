import time
import threading
import logging


def log_elapsed_time(process_name_getter):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            stop_flag = threading.Event()  # Event to signal the thread to stop
            process_name = process_name_getter(*args, **kwargs)
            start_time = time.time()
            logging.info(f"***Starting process: {process_name}")

            # Start the time tracker in a separate thread
            def time_tracker():
                while not stop_flag.is_set():  # Run until stop_flag is set
                    elapsed_time = time.time() - start_time
                    print(f"{process_name} Elapsed time: {int(elapsed_time)} seconds", end='\r')
                    time.sleep(1)
            
            tracker_thread = threading.Thread(target=time_tracker)
            tracker_thread.daemon = True
            tracker_thread.start()

            result = func(*args, **kwargs)  # Run the actual function

            stop_flag.set()  # Signal the tracker thread to stop
            tracker_thread.join()  # Wait for the tracker thread to finish

            elapsed_time = time.time() - start_time
            print(f"\n{process_name_getter(*args, **kwargs)} completed in {elapsed_time:.2f} seconds")
            return result

        return wrapper
    return decorator