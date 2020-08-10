### CS187 common code

##......................................................................
## Lightweight unit testing

class TimeoutError(Exception):
    pass

def timeout(func, args=(), kwargs={}, timeout_duration=1):
    '''Applies `func` to `args` and `kwargs`, returning the result, or 
       raising a `TimeoutError` exception if the application times out 
       after `timeout_duration` seconds. 
       (Based on code from https://stackoverflow.com/a/13821695)'''
    import signal

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)

    return result

def expect(func, args=(), kwargs={}, duration=5,
           expected_cb=lambda: "passed",
           failed_cb=lambda: "failed",
           timeout_cb=lambda duration: f"timed out after {duration}",
           exception_cb=lambda exc: f"failed with exception {exc}"):
    '''Applies `func` to `args` and `kwargs`, calling an appropriate callback 
       function depending on what occurs during the evaluation. The default 
       callbacks return an appropriate message as follows:
           result is True: "passed"
           result is False: "failed"
           exception raised: "failed with exception <exc>"
           timed out after `duration` seconds: "timed out after <duration>"
       '''
    try: 
        result = timeout(func, args, kwargs, timeout_duration=duration)
        return expected_cb() if result else failed_cb()
    except TimeoutError: 
        return timeout_cb(duration)
    except Exception as exc: 
        return exception_cb(exc)

## Unit test the unit test framework (!)

def main():
    '''Tests each of the four outcomes of `expect`. Should report all four 
       tests passed.'''

    def identity(x):
        return x

    def forever():
        while True:
            continue

    def raise_exc():
        return 3/0

    print("true test", 
          expect(lambda: "passed" == expect(identity, [True])))
    print("false test", 
          expect(lambda: "failed" == expect(identity, [False])))
    print("exception test", 
          expect(lambda: "failed with exception" == expect(raise_exc)[:21]))
    print("timeout test", 
          expect(lambda: "timed out" == expect(forever)[:9]))

if __name__ == "__main__":
    main()