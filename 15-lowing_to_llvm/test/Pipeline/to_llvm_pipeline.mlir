// RUN: export  NorthStarPassInclude=%ns-transform-include && ns-opt %s  --north-star-basic-pipeline="DP_Nums=2" | FileCheck %s

module @NorthStar {
  // CHECK: llvm.func @main
  // CHECK: llvm.call @softmax_1_128_softmax_1_128_fused_kernel
  // CHECK: llvm.call @softmax_1_128_softmax_1_128_fused_kernel
  func.func @main(%arg0: !north_star.ns_tensor<2x128xf32,0>) -> !north_star.ns_tensor<2x128xf32,0> attributes {dp_attr = #north_star.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "north_star.softmax"(%arg0) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.ns_tensor<2x128xf32,0>
    %1 = "north_star.softmax"(%0) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.ns_tensor<2x128xf32,0>
    return %1 : !north_star.ns_tensor<2x128xf32,0>
  }
}