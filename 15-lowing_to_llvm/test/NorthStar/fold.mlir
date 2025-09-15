// RUN: ns-opt %s  --split-input-file --inline | FileCheck %s

// CHECK-LABEL: NorthStar
// CHECK-NOT: north_star.add
module @NorthStar {
  func.func @main() -> !north_star.ns_tensor<2x2xf32,0> {
    %0 = "north_star.const"() <{value = dense<[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]> : !north_star.ns_tensor<2x2xf32,1>}> : () -> !north_star.ns_tensor<2x2xf32,0>
    %1 = "north_star.const"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : !north_star.ns_tensor<2x2xf32,1>}> : () -> !north_star.ns_tensor<2x2xf32,0>
    %2 = "north_star.add"(%0, %1) : (!north_star.ns_tensor<2x2xf32,0>, !north_star.ns_tensor<2x2xf32,0>) -> !north_star.ns_tensor<2x2xf32,0>
    return %2 : !north_star.ns_tensor<2x2xf32,0>
  }
}