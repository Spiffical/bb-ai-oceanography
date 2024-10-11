import time

class RateLimiter:
    def __init__(self, tokens, refresh_rate):
        self.tokens = tokens
        self.refresh_rate = refresh_rate
        self.last_refresh = time.time()

    def wait_for_token(self):
        while True:
            now = time.time()
            time_passed = now - self.last_refresh
            self.tokens = min(self.tokens + time_passed * self.refresh_rate, self.refresh_rate)
            self.last_refresh = now

            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                time.sleep(0.1)

# Initialize the rate limiter with 5 tokens per second
rate_limiter = RateLimiter(5, 5)

# Use the rate limiter before making any API call
def rate_limited_api_call(func, *args, **kwargs):
    rate_limiter.wait_for_token()
    return func(*args, **kwargs)