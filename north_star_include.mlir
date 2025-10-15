
module @NorthStar_include attributes { transform.with_named_sequence } {

transform.named_sequence @linalg_generalize(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module : (!transform.any_op) -> !transform.any_op
    %generalize_ops = transform.structured.generalize %linalg_ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @linalg_specialize(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module : (!transform.any_op) -> !transform.any_op
    %specialize_ops = transform.structured.specialize %linalg_ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @convert_elementwise_to_linalg(%module: !transform.any_op {transform.consumed}) ->(!transform.any_op) {
    %result = transform.apply_registered_pass "convert-elementwise-to-linalg" to %module: (!transform.any_op) -> !transform.any_op 
    transform.yield %result: !transform.any_op 
  }

transform.named_sequence @elementwise_fuse(%module: !transform.any_op {transform.consumed}) ->(!transform.any_op) {
    %result = transform.apply_registered_pass "linalg-fuse-elementwise-ops" to %module: (!transform.any_op) -> !transform.any_op 
    transform.yield %result: !transform.any_op 
  }

transform.named_sequence @linalg_analysis(%module: !transform.any_op {transform.consumed}) ->(!transform.any_op){
    %analysied_module = transform.bufferization.one_shot_bufferize
        layout{IdentityLayoutMap} %module {
          bufferize_function_boundaries=false,
          test_analysis_only= true,
          allow_unknown_ops = false,
          memcpy_op = "memref.copy" }
        : (!transform.any_op) -> !transform.any_op
    transform.yield %analysied_module: !transform.any_op 
  }


transform.named_sequence @linalg_decompose(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module : (!transform.any_op) -> !transform.any_op
    %decomposed_ops = transform.structured.decompose %linalg_ops : (!transform.any_op) -> !transform.any_op
    %softmax_ops = transform.structured.match ops{["linalg.softmax"]} in %module : (!transform.any_op) -> !transform.any_op
    %decomposed_softmax = transform.structured.decompose_interface %softmax_ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @flatten_elementwise(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module: (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %linalg_ops
      : (!transform.any_op) -> !transform.any_op
    %func_ops = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_ops {
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.op<"func.func">
    transform.yield
  }

transform.named_sequence @linalg_basic_fuse(%module: !transform.any_op {transform.consumed}) {
    transform.apply_patterns to %module {
        transform.apply_patterns.linalg.erase_unnecessary_inputs
      } : !transform.any_op
    %convert_elementwise_module = transform.include @convert_elementwise_to_linalg failures(suppress) (%module) : (!transform.any_op) -> (!transform.any_op )
    transform.include @linalg_specialize failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> ()
    transform.include @linalg_decompose failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> ()
    // // not run dynamic mode
    // // transform.include @flatten_elementwise failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> ()
    transform.include @linalg_generalize failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> ()
    %elementwise_fused_module = transform.include @elementwise_fuse failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> (!transform.any_op)
    transform.apply_patterns to %elementwise_fused_module {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
    // transform.include @linalg_analysis failures(suppress) (%elementwise_fused_module) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @linalg_basic_vectorization(%module: !transform.any_op {transform.readonly}) {
    transform.yield
  }

transform.named_sequence @finnal_bufferization(%module: !transform.any_op {transform.consumed}) {
    %result = transform.apply_registered_pass "func-bufferize" to %module: (!transform.any_op) -> !transform.any_op 
    transform.apply_patterns to %result {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
    %result_1 = transform.apply_registered_pass "finalizing-bufferize" to %result: (!transform.any_op) -> !transform.any_op 
    %res_to_para_module = transform.apply_registered_pass "buffer-results-to-out-params" to %result_1 {options = "hoist-static-allocs=true add-result-attr=true"}: (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @memref_basic_opt(%module: !transform.any_op {transform.readonly}) {    
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %add_deallocation_funcs = transform.apply_registered_pass "buffer-deallocation" to %funcs
    : (!transform.any_op) -> !transform.any_op
    %buffer_loop_hoisting = transform.apply_registered_pass "buffer-loop-hoisting" to %add_deallocation_funcs : (!transform.any_op) -> !transform.any_op
    %buffer_hoisting = transform.apply_registered_pass "buffer-hoisting" to %buffer_loop_hoisting : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %buffer_hoisting {
        // transform.apply_patterns.memref.alloc_to_alloca size_limit(128)
        transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims
        transform.apply_patterns.memref.expand_strided_metadata
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    %alloca = transform.structured.match ops{["memref.alloca"]} in %module
        : (!transform.any_op) -> !transform.op<"memref.alloca">
    %get_global, %global = transform.memref.alloca_to_global %alloca
          : (!transform.op<"memref.alloca">)
            -> (!transform.any_op, !transform.any_op)
    transform.memref.erase_dead_alloc_and_stores %buffer_hoisting : (!transform.any_op) -> ()
    // %lowing_affine_funcs = transform.apply_registered_pass "lower-affine" to %buffer_hoisting : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @lowing_to_llvm(%module: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %f = transform.apply_registered_pass "convert-vector-to-scf" to %func : (!transform.any_op) -> !transform.any_op
    %f3 = transform.apply_registered_pass "convert-scf-to-cf" to %f : (!transform.any_op) -> !transform.any_op
    %f4 = transform.apply_registered_pass "expand-strided-metadata" to %f3 : (!transform.any_op) -> !transform.any_op
    %f5 = transform.apply_registered_pass "lower-affine" to %f4 : (!transform.any_op) -> !transform.any_op

    transform.apply_conversion_patterns to %f5 {
      transform.apply_conversion_patterns.dialect_to_llvm "math"
      transform.apply_conversion_patterns.vector.vector_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "memref"
      transform.apply_conversion_patterns.func.func_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "index"
      transform.apply_conversion_patterns.dialect_to_llvm "arith"
      transform.apply_conversion_patterns.dialect_to_llvm "cf"
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        {index_bitwidth = 64,
        use_bare_ptr = false,
        use_bare_ptr_memref_call_conv = false,
        use_opaque_pointers = true}
    } {
      legal_dialects = ["llvm"],
      partial_conversion
    } : !transform.any_op

    %m2 = transform.apply_registered_pass "reconcile-unrealized-casts" to %module : (!transform.any_op) -> !transform.any_op
    transform.yield %m2 : !transform.any_op
  }

  transform.named_sequence @llvm_basic_opt(%module: !transform.any_op {transform.consumed}) {
    %legalized_module = transform.apply_registered_pass "llvm-legalize-for-export" to %module : (!transform.any_op) -> !transform.any_op 
    %alloca_opted_module = transform.apply_registered_pass "mem2reg" to %legalized_module : (!transform.any_op) -> !transform.any_op 
    transform.apply_dce to %alloca_opted_module : !transform.any_op
    transform.apply_cse to %alloca_opted_module : !transform.any_op
    transform.apply_patterns to %alloca_opted_module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  } 

} // transform module