# Logging

A small, thread-local logger with an on/off switch. There is no C++ standard logging library (unlike Python's `logging`); this header is the project's equivalent.

**Header:** `src/logging/logger.h`

```cpp
#include "logging/logger.h"
```

Logging is **off** by default. Calls to `info` / `warning` / `error` are no-ops until you enable it.

This header is **not** pulled in by `autograd/autograd.h` — include it yourself when you want logs.

---

## Enable and disable

The flag is thread-local (same pattern as `autograd::is_grad_enabled()`).

```cpp
bool logging::is_logging_enabled();          // false by default
void logging::set_logging_enabled(bool on);
```

```cpp
logging::set_logging_enabled(true);
logging::info("training started");
logging::set_logging_enabled(false);
```

Prefer the RAII guards below over toggling the flag by hand, so the previous mode is restored on every exit path (normal return or exception).

---

## RAII guards

`LoggingContext` remembers the current mode, switches to the requested mode, and restores the previous mode in its destructor.

```cpp
{
    logging::LoggingContext on(true);
    logging::info("visible");
}
// logging restored to whatever it was before the block
```

`NoLogContext` is a shorthand for `LoggingContext(false)`:

```cpp
logging::set_logging_enabled(true);

{
    logging::NoLogContext quiet;
    logging::info("silenced");   // no output
}

logging::info("visible again");
```

Guards nest. Each destructor restores the mode that was active when that guard was constructed:

```cpp
{
    logging::LoggingContext outer(true);
    {
        logging::LoggingContext inner(false);
        logging::info("silent");
    }
    logging::info("visible");    // outer still enabled
}
```

---

## Emit messages

| Function | Stream | Prefix |
|---|---|---|
| `logging::info(msg)` | `stdout` | `[INFO]` |
| `logging::warning(msg)` | `stderr` | `[WARNING]` |
| `logging::error(msg)` | `stderr` | `[ERROR]` |

```cpp
logging::LoggingContext on(true);

logging::info("epoch 1 done");
logging::warning("learning rate is very small");
logging::error("failed to load checkpoint");
```

Output:

```
[INFO] epoch 1 done
[WARNING] learning rate is very small
[ERROR] failed to load checkpoint
```

All three functions take a `const std::string&`. They return immediately when logging is disabled, so leaving `info(...)` calls in library code is cheap when the flag is off.

Library sites that emit when logging is on include `nn::save` / `nn::load`, non-strict `load_state_dict` warnings, and `contiguous()` packing (`copying tensor to contiguous storage…`).

---

## Example

```cpp
#include "autograd/autograd.h"
#include "logging/logger.h"

int main() {
    logging::LoggingContext logs(true);

    tensor::Tensor<float> w({2}, {0.1f, 0.2f}, true);
    tensor::Tensor<float> x({2}, {1.0f, 1.0f}, false);

    for (int step = 0; step < 10; ++step) {
        auto loss = ops::sum(ops::multiply(w, x));
        logging::info("step " + std::to_string(step));

        loss.backward();
        w.zero_grad();
    }
}
```

The runnable programs under `examples/` enable logging the same way. Training loops wrap the forward/backward pass in `NoLogContext` so library `contiguous()` messages stay off the step log. See [examples.md](examples.md).
