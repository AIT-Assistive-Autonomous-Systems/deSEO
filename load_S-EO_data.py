from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import socket
from urllib.error import URLError

# Define the retrying wrapper
@retry(
    stop=stop_after_attempt(5),                   # Try up to 5 times
    wait=wait_exponential(multiplier=2),          # Wait 2^x seconds between attempts
    retry=retry_if_exception_type((URLError, socket.error, ConnectionResetError)),  # Retry on network errors
    reraise=True                                   # Raise last exception if all retries fail
)
def load_with_retry():
    return load_dataset(
        "webdataset",
        data_files="datasets/SEO/rgb_crops.tar.gz",
        split="train"
    )

# Call the wrapped function
ds = load_with_retry()


