// -----// IR Dump After MarkDistributeParallelParametersPass (mark-distribute-parallel-parameters) //----- //
module @NorthStar {
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>, host_func} {
    %0 = "north_star.softmax"(%arg0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    %1 = "north_star.softmax"(%0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %1 : !north_star.ns_tensor<5x?x?xf32,0>
  }
}


