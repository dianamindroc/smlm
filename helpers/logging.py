import datetime
import os


def create_log_folder(log_dir, model_name):
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    if not isinstance(model_name, str):
        model_name = str(model_name)
    # Create the log folder name
    log_folder = f"logs_{model_name}_{timestamp}"

    # Create the log folder if it doesn't exist
    if not os.path.exists(os.path.join(log_dir,log_folder)):
        os.makedirs(os.path.join(log_dir,log_folder))

    return os.path.join(log_dir,log_folder)

