import time
from functools import wraps

from app.observability.langfuse_client import (
    langfuse
)


def traced_node(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        state = args[0] if args else {}

        question = state.get(
            "question",
            "unknown"
        )

        start = time.time()

        with langfuse.start_as_current_observation(
            as_type="span",
            name=func.__name__,
            input={
                "question": question
            }
        ) as span:

            result = func(
                *args,
                **kwargs
            )

            elapsed = round(
                time.time() - start,
                3
            )

            if isinstance(result, dict):
                node_timings = list(
                    state.get("node_timings", [])
                )

                node_timings.append(
                    {
                        "node": func.__name__,
                        "elapsed_seconds": elapsed,
                        "retry_count": state.get("retry_count", 0),
                    }
                )

                result = {
                    **result,
                    "node_timings": node_timings,
                }

            span.update(
                output=result,
                metadata={
                    "elapsed_seconds": elapsed,
                    "retry_count": state.get("retry_count", 0),
                    "node": func.__name__
                }

            )

            return result

    return wrapper
