module @NorthStar {
  func.func @main(%arg0: !north_star.ns_tensor<2x128xf32,0>) -> !north_star.ns_tensor<2x128xf32,0> attributes {dp_attr = #north_star.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "north_star.buffer"(%arg0) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    %1 = tensor.empty() {device_id = 0 : i64} : tensor<1x128xf32>
    %2 = "north_star.tensor_to_ns_tensor"(%1) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %3 = tensor.empty() {device_id = 1 : i64} : tensor<1x128xf32>
    %4 = "north_star.tensor_to_ns_tensor"(%3) <{device_id = 1 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,1>
    %5 = "north_star.buffer"(%2, %4) : (!north_star.ns_tensor<1x128xf32,0>, !north_star.ns_tensor<1x128xf32,1>) -> !north_star.buffer<0, 1>
    "north_star.scatter"(%0, %5) : (!north_star.buffer<0>, !north_star.buffer<0, 1>) -> ()
    %6 = "north_star.get_tensor"(%5) <{device_id = 0 : i64}> : (!north_star.buffer<0, 1>) -> !north_star.ns_tensor<1x128xf32,0>
    %7 = "north_star.ns_tensor_to_tensor"(%6) <{device_id = 0 : i64}> : (!north_star.ns_tensor<1x128xf32,0>) -> tensor<1x128xf32>
    %8 = "north_star.get_tensor"(%5) <{device_id = 1 : i64}> : (!north_star.buffer<0, 1>) -> !north_star.ns_tensor<1x128xf32,1>
    %9 = "north_star.ns_tensor_to_tensor"(%8) <{device_id = 1 : i64}> : (!north_star.ns_tensor<1x128xf32,1>) -> tensor<1x128xf32>
    %10 = call @softmax_1_128_softmax_1_128_fused_kernel(%7) {device_id = 0 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %11 = "north_star.tensor_to_ns_tensor"(%10) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %12 = call @softmax_1_128_softmax_1_128_fused_kernel(%9) {device_id = 1 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %13 = "north_star.tensor_to_ns_tensor"(%12) <{device_id = 1 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,1>
    %14 = "north_star.buffer"(%11, %13) : (!north_star.ns_tensor<1x128xf32,0>, !north_star.ns_tensor<1x128xf32,1>) -> !north_star.buffer<0, 1>
    %15 = tensor.empty() {device_id = 0 : i64} : tensor<2x128xf32>
    %16 = "north_star.tensor_to_ns_tensor"(%15) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %17 = "north_star.buffer"(%16) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    "north_star.gather"(%14, %17) : (!north_star.buffer<0, 1>, !north_star.buffer<0>) -> ()
    %18 = "north_star.get_tensor"(%17) <{device_id = 0 : i64}> : (!north_star.buffer<0>) -> !north_star.ns_tensor<2x128xf32,0>
    return %18 : !north_star.ns_tensor<2x128xf32,0>
  }
  func.func @softmax_1_128_softmax_1_128_fused_kernel(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> attributes {device_kernel} {
    %0 = tensor.empty() : tensor<1x128xf32>
    %1 = linalg.softmax dimension(1) ins(%arg0 : tensor<1x128xf32>) outs(%0 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %2 = tensor.empty() : tensor<1x128xf32>
    %3 = linalg.softmax dimension(1) ins(%1 : tensor<1x128xf32>) outs(%2 : tensor<1x128xf32>) -> tensor<1x128xf32>
    return %3 : tensor<1x128xf32>
  }
}