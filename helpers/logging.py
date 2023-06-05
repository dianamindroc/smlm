import datetime
import os


def create_log_folder(log_dir):
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Create the log folder name
    log_folder = f"logs_{timestamp}"

    # Create the log folder if it doesn't exist
    if not os.path.exists(os.path.join(log_dir,log_folder)):
        os.makedirs(log_folder)

    return log_folder

