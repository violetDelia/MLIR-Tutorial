module @NorthStar {
  func.func @main(%arg0: memref<2x128xf32>, %arg1: memref<2x128xf32>) {
    %0 = bufferization.to_tensor %arg0 : memref<2x128xf32>
    %1 = "north_star.tensor_to_ns_tensor"(%0) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %2 = "north_star.buffer"(%1) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %3 = bufferization.to_tensor %alloc : memref<1x128xf32>
    %4 = "north_star.tensor_to_ns_tensor"(%3) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %5 = bufferization.to_tensor %alloc_0 : memref<1x128xf32>
    %6 = "north_star.tensor_to_ns_tensor"(%5) <{device_id = 1 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,1>
    %7 = "north_star.buffer"(%4, %6) : (!north_star.ns_tensor<1x128xf32,0>, !north_star.ns_tensor<1x128xf32,1>) -> !north_star.buffer<0, 1>
    "north_star.scatter"(%2, %7) : (!north_star.buffer<0>, !north_star.buffer<0, 1>) -> ()
    %8 = "north_star.get_tensor"(%7) <{device_id = 0 : i64}> : (!north_star.buffer<0, 1>) -> !north_star.ns_tensor<1x128xf32,0>
    %9 = "north_star.ns_tensor_to_tensor"(%8) <{device_id = 0 : i64}> : (!north_star.ns_tensor<1x128xf32,0>) -> tensor<1x128xf32>
    %10 = bufferization.to_memref %9 : memref<1x128xf32>
    %11 = "north_star.get_tensor"(%7) <{device_id = 1 : i64}> : (!north_star.buffer<0, 1>) -> !north_star.ns_tensor<1x128xf32,1>
    %12 = "north_star.ns_tensor_to_tensor"(%11) <{device_id = 1 : i64}> : (!north_star.ns_tensor<1x128xf32,1>) -> tensor<1x128xf32>
    %13 = bufferization.to_memref %12 : memref<1x128xf32>
    %alloc_1 = memref.alloc() : memref<1x128xf32>
    call @softmax_1_128_softmax_1_128_fused_kernel(%10, %alloc_1) : (memref<1x128xf32>, memref<1x128xf32>) -> ()
    %14 = bufferization.to_tensor %alloc_1 : memref<1x128xf32>
    %15 = "north_star.tensor_to_ns_tensor"(%14) <{device_id = 0 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,0>
    %alloc_2 = memref.alloc() : memref<1x128xf32>
    call @softmax_1_128_softmax_1_128_fused_kernel(%13, %alloc_2) : (memref<1x128xf32>, memref<1x128xf32>) -> ()
    %16 = bufferization.to_tensor %alloc_2 : memref<1x128xf32>
    %17 = "north_star.tensor_to_ns_tensor"(%16) <{device_id = 1 : i64}> : (tensor<1x128xf32>) -> !north_star.ns_tensor<1x128xf32,1>
    %18 = "north_star.buffer"(%15, %17) : (!north_star.ns_tensor<1x128xf32,0>, !north_star.ns_tensor<1x128xf32,1>) -> !north_star.buffer<0, 1>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x128xf32>
    %19 = bufferization.to_tensor %alloc_3 : memref<2x128xf32>
    %20 = "north_star.tensor_to_ns_tensor"(%19) <{device_id = 0 : i64}> : (tensor<2x128xf32>) -> !north_star.ns_tensor<2x128xf32,0>
    %21 = "north_star.buffer"(%20) : (!north_star.ns_tensor<2x128xf32,0>) -> !north_star.buffer<0>
    "north_star.gather"(%18, %21) : (!north_star.buffer<0, 1>, !north_star.buffer<0>) -> ()
    %22 = "north_star.get_tensor"(%21) <{device_id = 0 : i64}> : (!north_star.buffer<0>) -> !north_star.ns_tensor<2x128xf32,0>
    %23 = "north_star.ns_tensor_to_tensor"(%22) <{device_id = 0 : i64}> : (!north_star.ns_tensor<2x128xf32,0>) -> tensor<2x128xf32>
    %24 = bufferization.to_memref %23 : memref<2x128xf32>
    memref.copy %24, %arg1 : memref<2x128xf32> to memref<2x128xf32>
    return
  }
  func.func @softmax_1_128_softmax_1_128_fused_kernel(%arg0: memref<1x128xf32>, %arg1: memref<1x128xf32>) attributes {device_kernel} {
    %0 = bufferization.to_tensor %arg0 : memref<1x128xf32>
    %1 = bufferization.to_memref %0 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.softmax dimension(1) ins(%1 : memref<1x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<1x128xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.softmax dimension(1) ins(%alloc : memref<1x128xf32>) outs(%alloc_0 : memref<1x128xf32>)
    %2 = bufferization.to_tensor %alloc_0 : memref<1x128xf32>
    %3 = bufferization.to_memref %2 : memref<1x128xf32>
    memref.copy %3, %arg1 : memref<1x128xf32> to memref<1x128xf32>
    return
  }
}

