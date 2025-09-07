# TVM Relay History Best - Migration Checklist

This document provides a step-by-step checklist for porting code between TVM versions when using Relay to retrieve and apply historical performance records.

## Migration from AutoTVM to MetaSchedule

### Pre-Migration Assessment

- [ ] **Identify Current AutoTVM Usage**
  - [ ] Locate `autotvm.apply_history_best()` calls
  - [ ] Find `.log` file dependencies  
  - [ ] Check for `autotvm.record.ApplyHistoryBest` usage
  - [ ] Note any custom AutoTVM database configurations

- [ ] **Check TVM Version Compatibility**
  - [ ] Verify current TVM version: `python -c "import tvm; print(tvm.__version__)"`
  - [ ] Confirm MetaSchedule availability: `python -c "import tvm.meta_schedule"`
  - [ ] Test database modules: `python -c "from tvm.meta_schedule import database"`

### Step 1: Database Migration

- [ ] **Convert Log Files to JSON Database**
  ```python
  # Old: AutoTVM log files
  log_file = "autotvm_tuning.log" 
  
  # New: Generate JSON database through re-tuning
  work_dir = "/path/to/new/database"
  # Run MetaSchedule tuning to generate:
  # - database_workload.json
  # - database_tuning_record.json
  ```

- [ ] **Verify Database Files Created**
  - [ ] Check `database_workload.json` exists and is valid JSON
  - [ ] Check `database_tuning_record.json` exists and is valid JSON
  - [ ] Verify file permissions and accessibility

### Step 2: Code Pattern Migration

- [ ] **Replace AutoTVM Context Manager**
  ```python
  # OLD PATTERN
  with autotvm.apply_history_best(log_file):
      with tvm.transform.PassContext(opt_level=3):
          lib = relay.build(mod, target=target)
  
  # NEW PATTERN  
  db = ms.database.JSONDatabase(workload_path, record_path)
  lib = ms.relay_integration.compile_relay(
      database=db, mod=mod, target=target, params=params
  )
  ```

- [ ] **Update Import Statements**
  ```python
  # Remove old imports
  # from tvm import autotvm
  # from tvm.autotvm.record import ApplyHistoryBest
  
  # Add new imports
  from tvm import meta_schedule as ms
  from tvm.ir.transform import PassContext
  ```

### Step 3: Compatibility Testing

- [ ] **Test Database Loading**
  ```python
  # Verify database loads successfully
  db = ms.database.JSONDatabase(workload_path, record_path)
  assert len(db) > 0, "Database should contain tuning records"
  ```

- [ ] **Test Compilation Pipeline**
  ```python
  # Ensure end-to-end compilation works
  lib = ms.relay_integration.compile_relay(
      database=db, mod=mod, target=target, params=params
  )
  assert lib is not None, "Compilation should succeed"
  ```

- [ ] **Validate Performance**
  - [ ] Run inference benchmarks with migrated code
  - [ ] Compare performance against original AutoTVM results
  - [ ] Document any performance regressions

## Migration Between MetaSchedule Versions

### From TVM 0.9.x to 0.10.x+

- [ ] **Update Database Creation Pattern**
  ```python
  # OLD: Manual database path construction
  db = ms.database.JSONDatabase(
      os.path.join(work_dir, "workloads.json"),
      os.path.join(work_dir, "tuning_records.json")
  )
  
  # NEW: Standard naming convention
  db = ms.database.JSONDatabase(
      work_dir + "/database_workload.json", 
      work_dir + "/database_tuning_record.json"
  )
  ```

- [ ] **Update Transform Pass Usage**
  ```python
  # OLD: Context manager approach
  with tvm.meta_schedule.ApplyHistoryBest(db):
      lib = relay.build(mod, target=target)
  
  # NEW: Transform pass approach  
  transform = relay.transform.MetaScheduleApplyHistoryBest(db)
  optimized_mod = transform(mod)
  lib = relay.build(optimized_mod, target=target)
  ```

### From Direct Transform to Integration API

- [ ] **Migrate to Unified API**
  ```python
  # OLD: Manual pass management
  with PassContext(opt_level=3, config={
      "relay.backend.use_meta_schedule": True
  }):
      pass_seq = tvm.transform.Sequential([
          relay.transform.MetaScheduleApplyHistoryBest(db)
      ])
      mod = pass_seq(mod)
      lib = relay.build(mod, target=target)
  
  # NEW: Integrated compilation
  lib = ms.relay_integration.compile_relay(
      database=db, mod=mod, target=target, params=params
  )
  ```

## Common Migration Issues and Solutions

### Issue: Database Format Incompatibility

- [ ] **Symptom**: `FileNotFoundError` or JSON parse errors
- [ ] **Solution**: Re-generate database with current TVM version
  ```python
  # Regenerate database
  with tempfile.TemporaryDirectory() as new_work_dir:
      db = ms.relay_integration.tune_relay(
          mod=mod, target=target, params=params,
          work_dir=new_work_dir, max_trials_global=100
      )
  ```

### Issue: Target Architecture Mismatch

- [ ] **Symptom**: Compilation succeeds but poor performance
- [ ] **Check**: Database target matches compilation target
  ```python
  # Verify target compatibility
  records = db.get_all_tuning_records()
  for record in records[:5]:  # Check first few records
      print(f"Record target: {record.target}")
      assert str(record.target) == str(target), "Target mismatch detected"
  ```

### Issue: ONNX Input Name Problems

- [ ] **Symptom**: Errors with input names containing special characters
- [ ] **Solution**: Sanitize input names before tuning
  ```python
  # Sanitize ONNX input names
  def sanitize_onnx_names(mod):
      # Replace problematic characters in input names
      # This should be done before both tuning and inference
      return relay.transform.InferType()(mod)
  
  mod = sanitize_onnx_names(mod)
  ```

### Issue: Shape Format Inconsistency  

- [ ] **Symptom**: Shape-related errors during compilation
- [ ] **Solution**: Ensure consistent shape representation
  ```python
  # Use consistent shape format
  input_shape = [1, 3, 224, 224]  # List format
  # Avoid: (1, 3, 224, 224)      # Tuple format in some contexts
  # Avoid: "1,3,224,224"         # String format
  ```

## Validation Checklist

### Pre-Deployment Testing

- [ ] **Functional Testing**
  - [ ] Model loads and compiles successfully
  - [ ] Inference produces correct outputs
  - [ ] Performance meets requirements

- [ ] **Integration Testing**  
  - [ ] Database loads in production environment
  - [ ] Memory usage within acceptable limits
  - [ ] Error handling works correctly

- [ ] **Performance Testing**
  - [ ] Benchmark inference latency
  - [ ] Measure compilation time
  - [ ] Profile memory consumption

### Post-Migration Monitoring

- [ ] **Performance Monitoring**
  - [ ] Set up inference latency tracking
  - [ ] Monitor memory usage patterns
  - [ ] Track compilation times

- [ ] **Error Monitoring**
  - [ ] Log database loading errors
  - [ ] Track compilation failures
  - [ ] Monitor inference errors

## Emergency Rollback Plan

- [ ] **Prepare Fallback Code**
  ```python
  def compile_with_fallback(mod, target, params, db_path=None):
      """Compile with database if available, fallback to basic compilation."""
      try:
          if db_path and os.path.exists(db_path):
              # Try MetaSchedule compilation
              db = load_database_flexible(db_path)
              return ms.relay_integration.compile_relay(
                  database=db, mod=mod, target=target, params=params
              )
      except Exception as e:
          print(f"Database compilation failed: {e}")
      
      # Fallback to basic compilation
      with tvm.transform.PassContext(opt_level=3):
          return relay.build(mod, target=target, params=params)
  ```

- [ ] **Document Rollback Procedure**
  - [ ] List steps to revert to previous version
  - [ ] Identify critical configuration files
  - [ ] Plan database backup and restoration

## Success Criteria

- [ ] **Migration is considered successful when:**
  - [ ] All tests pass with new implementation
  - [ ] Performance is within 5% of original
  - [ ] No new errors or warnings in logs
  - [ ] Database loading time is acceptable (<10s)
  - [ ] Memory usage increase is minimal (<10%)

- [ ] **Sign-off Requirements:**
  - [ ] Technical review completed
  - [ ] Performance benchmarks approved
  - [ ] Documentation updated
  - [ ] Training materials prepared for team