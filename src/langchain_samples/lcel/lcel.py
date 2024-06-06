""" Test some extra methods on runnable """
import random
from langchain_core.runnables import RunnableLambda


def add_one(x: int) -> int:
    """ add one """
    return x + 1


def buggy_double(y: int) -> int:
    """ Buggy double function """
    if random.random() > 0.3:
        print("This code failed, and will probably be retried")
        raise ValueError("Trigger buggy code")
    return y * 2


def use_retried_config():
    """ test Runnable.with_retry """
    sequence = (
        RunnableLambda(add_one) |
        RunnableLambda(buggy_double).with_retry(
            stop_after_attempt=10,
            wait_exponential_jitter=False
        )
    )
    print("input_schema=", sequence.input_schema.schema())
    print("output_schema=", sequence.output_schema.schema())
    print(sequence.invoke(1))
