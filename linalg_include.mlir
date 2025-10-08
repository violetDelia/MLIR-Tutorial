
module @linalg_include attributes { transform.with_named_sequence } {

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

// transform.named_sequence @vector_linalg(%func: !transform.op<"func.func"> {transform.consumed}) {
//     %generic_ops = transform.structured.match ops{["linalg.generic"]} in %func
//       : (!transform.op<"func.func">) -> !transform.any_op
//     transform.structured.vectorize %generic_ops vector_sizes [[4]] : !transform.any_op
//     transform.apply_patterns to %func {
//       transform.apply_patterns.vector.reduction_to_contract
//       transform.apply_patterns.vector.transfer_permutation_patterns
//     } : !transform.op<"func.func">

//     %func_h = transform.structured.hoist_redundant_vector_transfers %func
//       : (!transform.op<"func.func">) -> !transform.op<"func.func">
//     %all_loops = transform.structured.match interface{LoopLikeInterface} in %func_h
//       : (!transform.op<"func.func">) -> !transform.any_op
//     transform.apply_licm to %all_loops : !transform.any_op
//     transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

//     transform.apply_patterns to %func_h {
//       transform.apply_patterns.vector.reduction_to_contract
//       transform.apply_patterns.vector.cast_away_vector_leading_one_dim
//       transform.apply_patterns.canonicalization
//     } : !transform.op<"func.func">

//     transform.apply_patterns to %func_h {
//       transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
//       transform.apply_patterns.vector.lower_outerproduct
//     } : !transform.op<"func.func">
//     transform.yield
//   }

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
    // not run dynamic mode
    // transform.include @flatten_elementwise failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> ()
    transform.include @linalg_generalize failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> ()
    %elementwise_fused_module = transform.include @elementwise_fuse failures(suppress) (%convert_elementwise_module) : (!transform.any_op) -> (!transform.any_op)
    transform.apply_patterns to %elementwise_fused_module {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
    transform.include @linalg_analysis failures(suppress) (%elementwise_fused_module) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @linalg_basic_vectorization(%module: !transform.any_op {transform.readonly}) {
    transform.yield
  }

// transform.named_sequence @linalg_bufferization(%module: !transform.any_op {transform.consumed}) {
//     %empty_ops = transform.structured.match ops{["tensor.empty"]} in %module : (!transform.any_op) -> !transform.op<"tensor.empty">
//     transform.bufferization.empty_tensor_to_alloc_tensor %empty_ops : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
//     %bufferized_module = transform.bufferization.one_shot_bufferize
//         layout{IdentityLayoutMap} %module {
//           bufferize_function_boundaries=true,
//           test_analysis_only= false,
//           memcpy_op = "linalg.copy" }
//         : (!transform.any_op) -> !transform.any_op
//     %res_to_para_module = transform.apply_registered_pass "buffer-results-to-out-params" to %bufferized_module {options = "hoist-static-allocs=true add-result-attr=true"}: (!transform.any_op) -> !transform.any_op
//     // %alloc_to_arg_module = transform.apply_registered_pass "alloc-to-arg" to %res_to_para_module : (!transform.any_op) -> !transform.any_op
//     transform.include @linalg_analysis failures(suppress) (%res_to_para_module) : (!transform.any_op) -> !transform.any_op
//     transform.yield
//   }
} // transform module