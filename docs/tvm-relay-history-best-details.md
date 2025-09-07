# TVM Relay History Best - Detailed Documentation

This document provides detailed analysis of how Relay retrieves and applies historical performance records, with code excerpts from test files and API compatibility information.

## Test File Analysis

### test_meta_schedule_database.py

**Purpose**: Core MetaSchedule database functionality and JSONDatabase usage patterns

**Key Code Excerpt**:
```python
def _create_tmp_database(tmpdir: str, mod_eq: str = "structural") -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return ms.database.JSONDatabase(path_workload, path_tuning_record, module_equality=mod_eq)

# Usage example from test
with tempfile.TemporaryDirectory() as tmpdir:
    database = _create_tmp_database(tmpdir)
    workload = database.commit_workload(mod)
    record = ms.database.TuningRecord(
        _create_schedule(mod, _schedule_matmul).trace,
        workload,
        [1.5, 2.5, 1.8],
        tvm.target.Target("llvm"),
        ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
    )
    database.commit_tuning_record(record)
    
    # Reload database from files
    new_database = ms.database.JSONDatabase(
        path_workload=database.path_workload,
        path_tuning_record=database.path_tuning_record,
    )
```

**Explanation**: Demonstrates JSON database creation with separate workload and tuning record files. Shows how to persist and reload databases across sessions.

### test_meta_schedule_relay_integration.py

**Purpose**: End-to-end integration testing with relay compilation

**Key Code Excerpt**:
```python
# Extract tasks and tune
mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
extracted_tasks = ms.relay_integration.extract_tasks(mod, target="llvm", params=params)

# Database-driven compilation
with tempfile.TemporaryDirectory() as work_dir:
    database = ms.relay_integration.tune_relay(
        mod=mod,
        target=target,
        params=params,
        work_dir=work_dir,
        max_trials_global=4,
        strategy="replay-trace",
    )
    
    lib = ms.relay_integration.compile_relay(
        database=database,
        mod=mod,
        target=target, 
        params=params,
    )

# Verification database pattern
with target, _create_verification_database(), PassContext(
    opt_level=3,
    config={
        "relay.backend.use_meta_schedule": True,
        "relay.backend.use_meta_schedule_dispatch": 7,
        "relay.backend.tir_converter": "default",
    },
):
    rt_mod1 = relay.build(mod, target=target, params=params)
```

**Explanation**: Shows complete workflow from task extraction to database-driven compilation. The PassContext configuration enables MetaSchedule integration during relay.build().

### test_autotvm_record.py

**Purpose**: Legacy AutoTVM ApplyHistoryBest functionality

**Key Code Excerpt**:
```python
from tvm.autotvm.record import ApplyHistoryBest

# Load from file paths
hist_best = ApplyHistoryBest([file_path, file_path])
x = hist_best.query(target, tsk.workload)

# Load from Path objects
hist_best = ApplyHistoryBest(filepath_batch_1)
assert str(hist_best.query(target, tsk.workload)) == best

# Load from StringIO buffer
stringio_batch_1 = StringIO()
callback = autotvm.callback.log_to_file(stringio_batch_1)
callback(None, inputs_batch_1, results_batch_1)
stringio_batch_1.seek(0)
hist_best = ApplyHistoryBest(stringio_batch_1)

# Load from list of tuples
hist_best = ApplyHistoryBest(list(zip(inputs_batch_1, results_batch_1)))

# Multiple batches
hist_best = ApplyHistoryBest([filepath_batch_1, filepath_batch_2])
```

**Explanation**: Demonstrates flexible loading patterns for AutoTVM log data including files, streams, and in-memory data structures.

### test_micro_ms_tuning.py

**Purpose**: Modern MetaSchedule usage in real scenarios

**Key Code Excerpt**:
```python
# Database creation during tuning
db: ms.Database = ms.relay_integration.tune_relay(
    mod=mod,
    params=params,
    target=target,
    builder=builder,
    runner=runner,
    strategy="evolutionary",
    max_trials_per_task=10,
    max_trials_global=100,
    work_dir=str(work_dir),
    module_equality="ignore-ndarray",
)

# Compilation with database
ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
    database=db,
    mod=mod,
    target=target,
    params=params,
    pass_config=MappingProxyType({
        "relay.backend.use_meta_schedule": True,
        "relay.backend.tir_converter": "default",
        "tir.disable_vectorize": True,
    }),
    executor=executor,
    runtime=runtime,
)
```

**Explanation**: Shows production-ready MetaSchedule usage with proper pass configuration and database integration.

## API Compatibility Map

### Old vs New APIs

| **Old API** | **New API** | **Usage Context** |
|-------------|-------------|-------------------|
| `tvm.meta_schedule.ApplyHistoryBest(db)` | `relay.transform.MetaScheduleApplyHistoryBest(db)` | Context manager vs Transform pass |
| `autotvm.apply_history_best(log)` | `ms.relay_integration.compile_relay(database=db)` | Legacy log files vs JSON database |
| Manual PassContext setup | `ms.relay_integration.compile_relay()` | Manual vs integrated compilation |
| Log file loading | `ms.database.JSONDatabase(workload, records)` | Text logs vs structured JSON |

### Version Differences

**TVM 0.8.x and earlier:**
```python
# Old context manager pattern
with autotvm.apply_history_best(log_file):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
```

**TVM 0.9.x+:**
```python
# New transform pass pattern  
db = ms.database.JSONDatabase(workload_path, record_path)
with tvm.transform.PassContext(opt_level=3):
    pass_seq = tvm.transform.Sequential([
        relay.transform.MetaScheduleApplyHistoryBest(db),
        # other passes...
    ])
    mod = pass_seq(mod)
    lib = relay.build(mod, target=target)
```

**TVM 0.10.x+:**
```python
# Integrated compilation pattern
db = ms.database.JSONDatabase(workload_path, record_path)
lib = ms.relay_integration.compile_relay(
    database=db, mod=mod, target=target, params=params
)
```

## Ready-to-Use Shim Block

```python
import contextlib
import tvm
from tvm import relay, meta_schedule as ms

def apply_ms_best_or_fallback(mod, db_or_log):
    """
    Apply history best with version compatibility.
    
    Args:
        mod: IRModule to optimize
        db_or_log: Database object or log file path
        
    Returns:
        (modified_mod, context_manager)
    """
    # Try new MetaSchedule database approach
    if hasattr(db_or_log, 'query_tuning_record'):  # Database object
        try:
            # Try new transform pass
            if hasattr(relay.transform, "MetaScheduleApplyHistoryBest"):
                transform = relay.transform.MetaScheduleApplyHistoryBest(db_or_log)
                return transform(mod), contextlib.nullcontext()
        except Exception:
            pass
            
        try:
            # Try old context manager
            if hasattr(tvm.meta_schedule, "ApplyHistoryBest"):
                return mod, tvm.meta_schedule.ApplyHistoryBest(db_or_log)
        except Exception:
            pass
    
    # Try legacy AutoTVM approach
    if isinstance(db_or_log, (str, list)):  # Log file path(s)
        try:
            from tvm.autotvm.record import ApplyHistoryBest
            return mod, ApplyHistoryBest(db_or_log)
        except Exception:
            pass
    
    # Fallback: no optimization
    return mod, contextlib.nullcontext()

def load_database_flexible(work_dir_or_log):
    """
    Load database from various sources with fallback.
    
    Args:
        work_dir_or_log: Work directory, log file, or database
        
    Returns:
        Database object or log data
    """
    import os.path as osp
    
    # If already a database object, return as-is
    if hasattr(work_dir_or_log, 'query_tuning_record'):
        return work_dir_or_log
    
    # Try to load as MetaSchedule JSON database
    if isinstance(work_dir_or_log, str):
        if osp.isdir(work_dir_or_log):
            workload_path = osp.join(work_dir_or_log, "database_workload.json")
            record_path = osp.join(work_dir_or_log, "database_tuning_record.json")
            if osp.exists(workload_path) and osp.exists(record_path):
                return ms.database.JSONDatabase(workload_path, record_path)
        
        # Assume it's a log file path for AutoTVM
        return work_dir_or_log
    
    return work_dir_or_log

# Usage example:
# db = load_database_flexible("/path/to/work_dir")
# mod, ctx = apply_ms_best_or_fallback(mod, db)
# with ctx:
#     lib = relay.build(mod, target=target)
```

This shim provides maximum compatibility across TVM versions and handles both MetaSchedule databases and legacy AutoTVM logs.