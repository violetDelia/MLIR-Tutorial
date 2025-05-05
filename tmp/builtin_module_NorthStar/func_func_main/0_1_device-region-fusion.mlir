// -----// IR Dump After DeviceRegionFusionPass (device-region-fusion) //----- //
func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>, host_func} {
  %0:5 = "north_star.buffer_cast"(%arg0) <{distribute_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<1x?x?xf32,1>, !north_star.ns_tensor<1x?x?xf32,2>, !north_star.ns_tensor<1x?x?xf32,3>, !north_star.ns_tensor<1x?x?xf32,4>)
  %1 = call @softmax_1_d_d_softmax_1_d_d_0(%0#0) : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
  %2 = call @softmax_1_d_d_softmax_1_d_d_1(%0#1) : (!north_star.ns_tensor<1x?x?xf32,1>) -> !north_star.ns_tensor<1x?x?xf32,1>
  %3 = call @softmax_1_d_d_softmax_1_d_d_2(%0#2) : (!north_star.ns_tensor<1x?x?xf32,2>) -> !north_star.ns_tensor<1x?x?xf32,2>
  %4 = call @softmax_1_d_d_softmax_1_d_d_3(%0#3) : (!north_star.ns_tensor<1x?x?xf32,3>) -> !north_star.ns_tensor<1x?x?xf32,3>
  %5 = call @softmax_1_d_d_softmax_1_d_d_4(%0#4) : (!north_star.ns_tensor<1x?x?xf32,4>) -> !north_star.ns_tensor<1x?x?xf32,4>
  %6 = "north_star.buffer_cast"(%1, %2, %3, %4, %5) <{distribute_attr = #north_star.DP<DP = 5 : 0, 1, 2, 3, 4>}> : (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<1x?x?xf32,1>, !north_star.ns_tensor<1x?x?xf32,2>, !north_star.ns_tensor<1x?x?xf32,3>, !north_star.ns_tensor<1x?x?xf32,4>) -> !north_star.ns_tensor<5x?x?xf32,0>
  return %6 : !north_star.ns_tensor<5x?x?xf32,0>
}

