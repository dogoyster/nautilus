"""
TensorFlow bootstrap and runtime configuration helpers for Apple Silicon/macOS.

Goals:
- Avoid fork/mutex issues by forcing the 'spawn' start method early.
- Reduce threading contention that can surface as mutex errors.
- Optionally disable GPU/Metal if it causes instability.

Usage:
  from tf_compat import bootstrap, configure_tensorflow
  bootstrap()              # MUST be called before importing tensorflow or modules that import it
  import tensorflow as tf  # safe import after bootstrap
  configure_tensorflow()   # configure threads/devices after import
"""

import os
import platform
import multiprocessing as mp


def bootstrap() -> None:
    """Apply environment-level settings before TensorFlow is imported.

    This should be the first thing executed in any entrypoint using TensorFlow.
    """
    # Lower TF log verbosity
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Mac-specific: mitigate fork-safety initialization issues
    if platform.system() == "Darwin":
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    # OpenMP/MKL duplicate symbol guard that often manifests as a mutex/lock error
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Keep thread counts small to avoid contention by default
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Ensure safe multiprocessing semantics with libraries that aren't fork-safe
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method was already set by the process - ignore
        pass


def configure_tensorflow(disable_gpu_default: bool = True) -> None:
    """Apply TensorFlow runtime configuration after TF import.

    - Constrains intra/inter op threads to 1
    - Optionally disables GPU/Metal (can be toggled via NAUTILUS_TF_DISABLE_GPU)
    """
    try:
        import tensorflow as tf  # noqa: WPS433 (runtime import by design)

        # Reduce thread contention
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        # Optionally disable GPU/Metal which can trigger driver-related mutex issues on some setups
        should_disable_gpu = os.environ.get("NAUTILUS_TF_DISABLE_GPU")
        if should_disable_gpu is None:
            disable_gpu = disable_gpu_default
        else:
            disable_gpu = should_disable_gpu == "1"

        if disable_gpu:
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
    except Exception:
        # If TF isn't installed or import fails, just skip runtime config
        pass


