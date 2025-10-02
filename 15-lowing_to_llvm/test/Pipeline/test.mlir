module @NorthStar {
  func.func @main(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = "north_star.tensor_to_ns_tensor"(%arg0) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %1 = "north_star.buffer"(%0) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    %2 = tensor.empty() {device_id = 0 : i64} : tensor<1x128xf32>
    %3 = "north_star.tensor_to_ns_tensor"(%2) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %4 = tensor.empty() {device_id = 1 : i64} : tensor<1x128xf32>
    %5 = "north_star.tensor_to_ns_tensor"(%4) <{device_id = 1 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,1>
    %6 = "north_star.buffer"(%3, %5) : (!north_star.ns_tensor<1x128xf32,0>, !north_star.ns_tensor<1x128xf32,1>) -> !north_star.buffer<0, 1>
    "north_star.scatter"(%1, %6) : (!north_star.buffer<0>, !north_star.buffer<0, 1>) -> ()
    %7 = "north_star.get_tensor"(%6) <{device_id = 0 : i64}> : (!north_star.buffer<0, 1>) -> !north_star.ns_tensor<1x128xf32,0>
    %8 = "north_star.ns_tensor_to_tensor"(%7) <{device_id = 0 : i64}> : (!north_star.ns_tensor<1x128xf32,0>) -> tensor<1x128xf32>
    %9 = "north_star.get_tensor"(%6) <{device_id = 1 : i64}> : (!north_star.buffer<0, 1>) -> !north_star.ns_tensor<1x128xf32,1>
    %10 = "north_star.ns_tensor_to_tensor"(%9) <{device_id = 1 : i64}> : (!north_star.ns_tensor<1x128xf32,1>) -> tensor<1x128xf32>
    %11 = call @softmax_1_128_softmax_1_128_fused_kernel(%8) {device_id = 0 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %12 = "north_star.tensor_to_ns_tensor"(%11) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %13 = call @softmax_1_128_softmax_1_128_fused_kernel(%10) {device_id = 1 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %14 = "north_star.tensor_to_ns_tensor"(%13) <{device_id = 1 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,1>
    %15 = "north_star.buffer"(%12, %14) : (!north_star.ns_tensor<1x128xf32,0>, !north_star.ns_tensor<1x128xf32,1>) -> !north_star.buffer<0, 1>
    %16 = tensor.empty() {device_id = 0 : i64} : tensor<2x128xf32>
    %17 = "north_star.tensor_to_ns_tensor"(%16) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %18 = "north_star.buffer"(%17) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    "north_star.gather"(%15, %18) : (!north_star.buffer<0, 1>, !north_star.buffer<0>) -> ()
    %19 = "north_star.get_tensor"(%18) <{device_id = 0 : i64}> : (!north_star.buffer<0>) -> !north_star.ns_tensor<2x128xf32,0>
    %20 = "north_star.ns_tensor_to_tensor"(%19) <{device_id = 0 : i64}> : (!north_star.ns_tensor<2x128xf32,0>) -> tensor<2x128xf32>
    return %20 : tensor<2x128xf32>
  }
  func.func @softmax_1_128_softmax_1_128_fused_kernel(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> attributes {device_kernel} {
    %0 = tensor.empty() : tensor<1x128xf32>
    %1 = linalg.softmax dimension(1) ins(%arg0 : tensor<1x128xf32>) outs(%0 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %2 = tensor.empty() : tensor<1x128xf32>
    %3 = linalg.softmax dimension(1) ins(%1 : tensor<1x128xf32>) outs(%2 : tensor<1x128xf32>) -> tensor<1x128xf32>
    return %3 : tensor<1x128xf32>
  }
}

// '/home/lfr/MLIR_Tutorial/build/third_party/llvm-project/llvm/bin/mlir-opt' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=fully-dynamic-layout-map"  -allow-unregistered-dialect -func-bufferize -buffer-results-to-out-params="hoist-static-allocs" -o '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test_out.mlir' 