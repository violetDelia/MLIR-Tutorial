// RUN: ns-opt %s --mark-distribute-parallel-parameters="DP=5 TP=1" | FileCheck %s

module @NorthStar {
  // CHECK-LABEL:   func @main(
  // CHECK-SAME: #north_star.DP<DP = 5
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "north_star.softmax"(%arg0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    %1 = "north_star.softmax"(%0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %1 : !north_star.ns_tensor<5x?x?xf32,0>
  }
}