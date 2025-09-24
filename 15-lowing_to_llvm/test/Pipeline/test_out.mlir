module @NorthStar {
  func.func @main(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = "north_star.tensor_to_ns_tensor"(%arg0) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %1 = "north_star.buffer"(%0) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %2 = bufferization.to_tensor %alloc : memref<1x128xf32>
    %3 = "north_star.tensor_to_ns_tensor"(%2) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %4 = bufferization.to_tensor %alloc_0 : memref<1x128xf32>
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
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x128xf32>
    %16 = bufferization.to_tensor %alloc_1 : memref<2x128xf32>
    %17 = "north_star.tensor_to_ns_tensor"(%16) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %18 = "north_star.buffer"(%17) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    "north_star.gather"(%15, %18) : (!north_star.buffer<0, 1>, !north_star.buffer<0>) -> ()
    %19 = "north_star.get_tensor"(%18) <{device_id = 0 : i64}> : (!north_star.buffer<0>) -> !north_star.ns_tensor<2x128xf32,0>
    %20 = "north_star.ns_tensor_to_tensor"(%19) <{device_id = 0 : i64}> : (!north_star.ns_tensor<2x128xf32,0>) -> tensor<2x128xf32>
    return %20 : tensor<2x128xf32>
  }
  func.func @softmax_1_128_softmax_1_128_fused_kernel(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> attributes {device_kernel} {
    %0 = bufferization.to_memref %arg0 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.softmax dimension(1) ins(%0 : memref<1x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<1x128xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.softmax dimension(1) ins(%alloc : memref<1x128xf32>) outs(%alloc_0 : memref<1x128xf32>)
    %1 = bufferization.to_tensor %alloc_0 : memref<1x128xf32>
    return %1 : tensor<1x128xf32>
  }
}

