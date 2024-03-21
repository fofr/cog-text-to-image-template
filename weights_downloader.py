import subprocess
import time
import os


class WeightsDownloader:
    def __init__(self):
        pass

    def download_if_not_exists(self, weight_str, url, dest):
        if not os.path.exists(f"{dest}/{weight_str}"):
            self.download(weight_str, url, dest)

    def download(self, weight_str, url, dest):
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weight_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"⌛️ Downloaded {weight_str} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"⌛️ Downloaded {weight_str} in {elapsed_time:.2f}s")
