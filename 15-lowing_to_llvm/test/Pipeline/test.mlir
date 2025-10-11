#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module @NorthStar {
  func.func @main(%arg0: tensor<2x128xf32> {func.device_id = 0 : i64}) -> tensor<2x128xf32> {
    %0 = bufferization.to_memref %arg0 : memref<2x128xf32, strided<[?, ?], offset: ?>>
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x128xf32>
    memref.copy %0, %alloc : memref<2x128xf32, strided<[?, ?], offset: ?>> to memref<2x128xf32>
    %cast = memref.cast %alloc : memref<2x128xf32> to memref<*xf32>
    %3 = bufferization.to_tensor %cast : memref<*xf32>
    %4 = call @__NS__MemrefToNSMemref_f32(%c0_i64, %3) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %5 = llvm.alloca %2 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %6 = llvm.alloca %2 x i64 : (i64) -> !llvm.ptr
    llvm.store %4, %5 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    llvm.store %c0_i64, %6 : i64, !llvm.ptr
    %7 = call @__NS__MakeBuffer_f32(%5, %6, %2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__SetDevice(%c0_i64) : (i64) -> ()
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %cast_1 = memref.cast %alloc_0 : memref<1x128xf32> to memref<*xf32>
    %8 = bufferization.to_tensor %cast_1 : memref<*xf32>
    %9 = call @__NS__MemrefToNSMemref_f32(%c0_i64, %8) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    call @__NS__SetDevice(%c1_i64) : (i64) -> ()
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %cast_3 = memref.cast %alloc_2 : memref<1x128xf32> to memref<*xf32>
    %10 = bufferization.to_tensor %cast_3 : memref<*xf32>
    %11 = call @__NS__MemrefToNSMemref_f32(%c1_i64, %10) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %12 = llvm.alloca %1 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %13 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
    llvm.store %9, %12 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    llvm.store %c0_i64, %13 : i64, !llvm.ptr
    %14 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %11, %14 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %15 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %c1_i64, %15 : i64, !llvm.ptr
    %16 = call @__NS__MakeBuffer_f32(%12, %13, %1) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__Scatter(%7, %16) : (!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) -> ()
    %17 = call @__NS__GetTensor_f32(%c0_i64, %16) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %18 = call @__NS__NSMemrefToMemref_f32(%17) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
    %19 = bufferization.to_memref %18 : memref<*xf32>
    %cast_4 = memref.cast %19 : memref<*xf32> to memref<1x128xf32, strided<[?, ?], offset: ?>>
    %20 = bufferization.to_tensor %cast_4 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %21 = call @__NS__GetTensor_f32(%c1_i64, %16) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %22 = call @__NS__NSMemrefToMemref_f32(%21) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
    %23 = bufferization.to_memref %22 : memref<*xf32>
    %cast_5 = memref.cast %23 : memref<*xf32> to memref<1x128xf32, strided<[?, ?], offset: ?>>
    %24 = bufferization.to_tensor %cast_5 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    call @__NS__SetDevice(%c0_i64) : (i64) -> ()
    %25 = call @softmax_1_128_softmax_1_128_fused_kernel(%20) {device_id = 0 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %26 = bufferization.to_memref %25 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    memref.copy %26, %alloc_6 : memref<1x128xf32, strided<[?, ?], offset: ?>> to memref<1x128xf32>
    %cast_7 = memref.cast %alloc_6 : memref<1x128xf32> to memref<*xf32>
    %27 = bufferization.to_tensor %cast_7 : memref<*xf32>
    %28 = call @__NS__MemrefToNSMemref_f32(%c0_i64, %27) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    call @__NS__SetDevice(%c1_i64) : (i64) -> ()
    %29 = call @softmax_1_128_softmax_1_128_fused_kernel(%24) {device_id = 1 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %30 = bufferization.to_memref %29 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    memref.copy %30, %alloc_8 : memref<1x128xf32, strided<[?, ?], offset: ?>> to memref<1x128xf32>
    %cast_9 = memref.cast %alloc_8 : memref<1x128xf32> to memref<*xf32>
    %31 = bufferization.to_tensor %cast_9 : memref<*xf32>
    %32 = call @__NS__MemrefToNSMemref_f32(%c1_i64, %31) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %33 = llvm.alloca %1 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %34 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
    llvm.store %28, %33 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    llvm.store %c0_i64, %34 : i64, !llvm.ptr
    %35 = llvm.getelementptr %33[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %32, %35 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %36 = llvm.getelementptr %34[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %c1_i64, %36 : i64, !llvm.ptr
    %37 = call @__NS__MakeBuffer_f32(%33, %34, %1) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__SetDevice(%c0_i64) : (i64) -> ()
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<2x128xf32>
    %cast_11 = memref.cast %alloc_10 : memref<2x128xf32> to memref<*xf32>
    %38 = bufferization.to_tensor %cast_11 : memref<*xf32>
    %39 = call @__NS__MemrefToNSMemref_f32(%c0_i64, %38) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %40 = llvm.alloca %2 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %41 = llvm.alloca %2 x i64 : (i64) -> !llvm.ptr
    llvm.store %39, %40 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    llvm.store %c0_i64, %41 : i64, !llvm.ptr
    %42 = call @__NS__MakeBuffer_f32(%40, %41, %2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__Gather(%37, %42) : (!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) -> ()
    %43 = call @__NS__GetTensor_f32(%c0_i64, %42) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %44 = call @__NS__NSMemrefToMemref_f32(%43) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
    %45 = bufferization.to_memref %44 : memref<*xf32>
    %cast_12 = memref.cast %45 : memref<*xf32> to memref<2x128xf32, strided<[?, ?], offset: ?>>
    %46 = bufferization.to_tensor %cast_12 : memref<2x128xf32, strided<[?, ?], offset: ?>>
    return %46 : tensor<2x128xf32>
  }
  func.func private @__NS__SetDevice(i64)
  func.func private @softmax_1_128_softmax_1_128_fused_kernel(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> attributes {device_kernel} {
    %0 = bufferization.to_memref %arg0 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : f32) outs(%alloc_1 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%1 : memref<1x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc_1 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maxnumf %in, %out : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%0, %alloc_1 : memref<1x128xf32, strided<[?, ?], offset: ?>>, memref<1xf32>) outs(%alloc : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %3 = arith.subf %in, %in_4 : f32
      %4 = math.exp %3 : f32
      linalg.yield %4 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst : f32) outs(%alloc_1 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<1x128xf32>) outs(%alloc_1 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_1 : memref<1x128xf32>, memref<1xf32>) outs(%alloc : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %3 = arith.divf %in, %in_4 : f32
      linalg.yield %3 : f32
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : f32) outs(%alloc_3 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<1x128xf32>) outs(%alloc_3 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maxnumf %in, %out : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_3 : memref<1x128xf32>, memref<1xf32>) outs(%alloc_2 : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %3 = arith.subf %in, %in_4 : f32
      %4 = math.exp %3 : f32
      linalg.yield %4 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst : f32) outs(%alloc_3 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%alloc_2 : memref<1x128xf32>) outs(%alloc_3 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc_2, %alloc_3 : memref<1x128xf32>, memref<1xf32>) outs(%alloc_2 : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %3 = arith.divf %in, %in_4 : f32
      linalg.yield %3 : f32
    }
    %2 = bufferization.to_tensor %alloc_2 : memref<1x128xf32>
    return %2 : tensor<1x128xf32>
  }
  func.func private @__NS__MemrefToNSMemref_f32(i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
  func.func private @__NS__MakeBuffer_f32(!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
  func.func private @__NS__Scatter(!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>)
  func.func private @__NS__GetTensor_f32(i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
  func.func private @__NS__NSMemrefToMemref_f32(!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
  func.func private @__NS__Gather(!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>)
}

// '/home/lfr/MLIR_Tutorial/build/third_party/llvm-project/llvm/bin/mlir-opt' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' -func-bufferize  -canonicalize -finalizing-bufferize  --mlir-print-ir-after-all -convert-linalg-to-affine-loops -buffer-results-to-out-params="hoist-static-allocs" --split-input-file -o '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test_out.mlir' &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'

// '/home/lfr/MLIR_Tutorial/build/15-lowing_to_llvm/src/Tools/NS-opt/NS-opt15' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' --convert-north-satr-to-func --mlir-print-ir-after-all -o '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test_out.mlir' --debug &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'

// '/home/lfr/MLIR_Tutorial/build/15-lowing_to_llvm/src/Tools/NS-opt/NS-opt15' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/to_llvm_pipeline.mlir' --north-star-basic-pipeline="DP_Nums=2" --mlir-print-ir-after-all   --debug &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'

// '/home/lfr/MLIR_Tutorial/build/15-lowing_to_llvm/src/Tools/NS-opt/NS-opt15' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' --north-star-basic-pipeline="DP_Nums=2" --mlir-print-ir-after-all --debug &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'