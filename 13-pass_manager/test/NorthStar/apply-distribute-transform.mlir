// RUN: ns-opt %s --apply-distribute-transform  --split-input-file | FileCheck %s

module @NorthStar {
  // CHECK-LABEL: func @main(
  // CHECK-COUNT-4: north_star.softmax
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "north_star.softmax"(%arg0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    %1 = "north_star.softmax"(%0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %1 : !north_star.ns_tensor<5x?x?xf32,0>
  }
}

// -----
module @NorthStar {
  // CHECK-LABEL: func @main(
  // CHECK-COUNT-3: north_star.softmax
  // CHECK-NEXT: north_star.buffer_cast
  // CHECK-NEXT: north_star.buffer_cast
  // CHECK-COUNT-3: north_star.softmax
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0 = "north_star.softmax"(%arg0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    %1 = "north_star.softmax"(%0) <{axis = 1 : i64}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %1 : !north_star.ns_tensor<5x?x?xf32,0>
  }
}