# TVM Relay History Best - Index

This document provides an index of relevant test files and quick-start snippets for using Relay to retrieve and apply historical performance records from MetaSchedule and AutoTVM tuning databases.

## Relevant Test Files

### MetaSchedule Database Tests
- **`tests/python/unittest/test_meta_schedule_database.py`** - Core database functionality tests
  - JSONDatabase creation and usage patterns
  - Loading from `database_workload.json` and `database_tuning_record.json`
  - Database query operations and top-k retrieval

### MetaSchedule Relay Integration Tests  
- **`tests/python/unittest/test_meta_schedule_relay_integration.py`** - End-to-end integration
  - Task extraction and tuning workflows
  - Database compilation with `ms.relay_integration.compile_relay()`
  - PassContext usage patterns

### AutoTVM Legacy Tests
- **`tests/python/unittest/test_autotvm_record.py`** - Legacy AutoTVM functionality
  - `ApplyHistoryBest` usage patterns
  - Loading from log files and applying best records
  - Multiple file and data source handling

### Micro Tuning Tests
- **`tests/python/unittest/test_micro_ms_tuning.py`** - Modern MetaSchedule usage
  - Database integration in real tuning scenarios
  - Usage with `relay.build()` and PassContext

## Quick-Start Snippet

```python
import tvm
from tvm import relay, meta_schedule as ms
from tvm.ir.transform import PassContext
import contextlib

# Load MetaSchedule database 
work_dir = "/path/to/tuning/results"
db = ms.database.JSONDatabase(
    work_dir + "/database_workload.json",
    work_dir + "/database_tuning_record.json"
)

# Apply history best (compatibility shim)
def apply_ms_best_or_fallback(mod, db):
    try:
        if hasattr(tvm.meta_schedule, "ApplyHistoryBest"):
            return mod, tvm.meta_schedule.ApplyHistoryBest(db)
    except: pass
    for c in [getattr(relay.transform,"MetaScheduleApplyHistoryBest",None),
              getattr(relay.transform,"ApplyHistoryBest",None)]:
        if c: return c(db)(mod), contextlib.nullcontext()
    return mod, contextlib.nullcontext()

# Example usage
mod = tvm.IRModule.from_expr(relay_func)
target = "llvm --num-cores=4"

# Apply database and build
tuned_mod, ctx = apply_ms_best_or_fallback(mod, db)
with ctx, PassContext(opt_level=3):
    lib = relay.build(tuned_mod, target=target, params=params)

# Alternative: Use relay integration
lib = ms.relay_integration.compile_relay(
    database=db, mod=mod, target=target, params=params
)
```

## Key Concepts

- **JSONDatabase**: Loads tuning records from JSON files
- **ApplyHistoryBest**: Retrieves and applies best performance records  
- **Database vs Log Files**: MetaSchedule uses structured JSON, AutoTVM uses log files
- **PassContext**: Controls optimization passes and database application
- **Version Compatibility**: New transform passes vs old context managers

## Common Pitfalls

1. **ONNX Input Names**: Names with `:` characters may cause issues
2. **Shape Format**: Ensure consistent shape format (`1,3,224,224`)
3. **Target Mismatch**: Database target must match current compilation target
4. **File Paths**: Use absolute paths for database files
5. **API Version**: Different TVM versions have different API patterns

See detailed documentation for comprehensive examples and migration guidance.