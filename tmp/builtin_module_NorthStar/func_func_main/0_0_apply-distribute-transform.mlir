// -----// IR Dump After ApplyDistributeTransformPass (apply-distribute-transform) //----- //
func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>, host_func} {
  %0:5 = "north_star.buffer_cast"(%arg0) <{distribute_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<1x?x?xf32,1>, !north_star.ns_tensor<1x?x?xf32,2>, !north_star.ns_tensor<1x?x?xf32,3>, !north_star.ns_tensor<1x?x?xf32,4>)
  %1 = "north_star.softmax"(%0#0) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
  %2 = "north_star.softmax"(%0#1) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,1>) -> !north_star.ns_tensor<1x?x?xf32,1>
  %3 = "north_star.softmax"(%0#2) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,2>) -> !north_star.ns_tensor<1x?x?xf32,2>
  %4 = "north_star.softmax"(%0#3) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,3>) -> !north_star.ns_tensor<1x?x?xf32,3>
  %5 = "north_star.softmax"(%0#4) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,4>) -> !north_star.ns_tensor<1x?x?xf32,4>
  %6 = "north_star.buffer_cast"(%1, %2, %3, %4, %5) <{distribute_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>}> : (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<1x?x?xf32,1>, !north_star.ns_tensor<1x?x?xf32,2>, !north_star.ns_tensor<1x?x?xf32,3>, !north_star.ns_tensor<1x?x?xf32,4>) -> !north_star.ns_tensor<5x?x?xf32,0>
  %7:5 = "north_star.buffer_cast"(%6) <{distribute_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<1x?x?xf32,1>, !north_star.ns_tensor<1x?x?xf32,2>, !north_star.ns_tensor<1x?x?xf32,3>, !north_star.ns_tensor<1x?x?xf32,4>)
  %8 = "north_star.softmax"(%7#0) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
  %9 = "north_star.softmax"(%7#1) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,1>) -> !north_star.ns_tensor<1x?x?xf32,1>
  %10 = "north_star.softmax"(%7#2) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,2>) -> !north_star.ns_tensor<1x?x?xf32,2>
  %11 = "north_star.softmax"(%7#3) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,3>) -> !north_star.ns_tensor<1x?x?xf32,3>
  %12 = "north_star.softmax"(%7#4) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,4>) -> !north_star.ns_tensor<1x?x?xf32,4>
  %13 = "north_star.buffer_cast"(%8, %9, %10, %11, %12) <{distribute_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>}> : (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<1x?x?xf32,1>, !north_star.ns_tensor<1x?x?xf32,2>, !north_star.ns_tensor<1x?x?xf32,3>, !north_star.ns_tensor<1x?x?xf32,4>) -> !north_star.ns_tensor<5x?x?xf32,0>
  return %13 : !north_star.ns_tensor<5x?x?xf32,0>
}

