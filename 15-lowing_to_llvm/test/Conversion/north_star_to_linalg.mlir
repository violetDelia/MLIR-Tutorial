// RUN: ns-opt %s --convert-north-satr-to-linalg  --reconcile-unrealized-casts --split-input-file | FileCheck %s
module @NorthStar {
  // CHECK-COUNT-2: north_star.softmax
  // CHECK-COUNT-4: linalg.softmax
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0:3 = "north_star.buffer_cast"(%arg0) <{distribute_attr = #north_star.DP<DP = 3 : 0, 1, 2>}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<2x?x?xf32,1>, !north_star.ns_tensor<2x?x?xf32,2>)
    %1 = "north_star.softmax"(%0#0) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
    %6 = "north_star.softmax"(%1) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
    %2 = "north_star.device_kernel"(%0#1) <{device_id = 1 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !north_star.ns_tensor<2x?x?xf32,1>):
      %52 = "north_star.softmax"(%arg1) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,1>) -> !north_star.ns_tensor<2x?x?xf32,1>
      %62 = "north_star.softmax"(%52) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,1>) -> !north_star.ns_tensor<2x?x?xf32,1>
      north_star.return %62 : !north_star.ns_tensor<2x?x?xf32,1>
    }) : (!north_star.ns_tensor<2x?x?xf32,1>) -> !north_star.ns_tensor<2x?x?xf32,1>
    %3 = "north_star.device_kernel"(%0#2) <{device_id = 2 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !north_star.ns_tensor<2x?x?xf32,2>):
      %53 = "north_star.softmax"(%arg1) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<2x?x?xf32,2>
      %63 = "north_star.softmax"(%53) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<2x?x?xf32,2>
      north_star.return %63 : !north_star.ns_tensor<2x?x?xf32,2>
    }) : (!north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<2x?x?xf32,2>
    %4 = "north_star.buffer_cast"(%6, %2, %3) <{distribute_attr = #north_star.DP<DP = 3 : 0, 1, 2>}> : (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<2x?x?xf32,1>, !north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %4 : !north_star.ns_tensor<5x?x?xf32,0>
  }
}