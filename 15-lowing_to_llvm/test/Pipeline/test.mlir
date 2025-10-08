module @NorthStar {
  func.func @main(%arg0: tensor<2x128xf32> {func.device_id = 0 : i64}) -> tensor<2x128xf32> {
    %0 = bufferization.to_memref %arg0 : memref<2x128xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x128xf32>
    memref.copy %0, %alloc : memref<2x128xf32, strided<[?, ?], offset: ?>> to memref<2x128xf32>
    %cast = memref.cast %alloc : memref<2x128xf32> to memref<*xf32>
    %1 = bufferization.to_tensor %cast : memref<*xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0_i64_0 = arith.constant 0 : i64
    %2 = call @__NS__MemrefToNSMemref_f32(%c0_i64_0, %1) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %5 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %6 = llvm.getelementptr %4[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %2, %6 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %7 = llvm.getelementptr %5[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %c0_i64_1 = arith.constant 0 : i64
    llvm.store %c0_i64_1, %7 : i64, !llvm.ptr
    %8 = call @__NS__MakeBuffer_f32(%4, %5, %3) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__SetDevice(%c0_i64) : (i64) -> ()
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %cast_3 = memref.cast %alloc_2 : memref<1x128xf32> to memref<*xf32>
    %9 = bufferization.to_tensor %cast_3 : memref<*xf32>
    %c0_i64_4 = arith.constant 0 : i64
    %10 = call @__NS__MemrefToNSMemref_f32(%c0_i64_4, %9) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    call @__NS__SetDevice(%c1_i64) : (i64) -> ()
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    %cast_6 = memref.cast %alloc_5 : memref<1x128xf32> to memref<*xf32>
    %11 = bufferization.to_tensor %cast_6 : memref<*xf32>
    %c1_i64_7 = arith.constant 1 : i64
    %12 = call @__NS__MemrefToNSMemref_f32(%c1_i64_7, %11) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %13 = llvm.mlir.constant(2 : i64) : i64
    %14 = llvm.alloca %13 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %15 = llvm.alloca %13 x i64 : (i64) -> !llvm.ptr
    %16 = llvm.getelementptr %14[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %10, %16 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %17 = llvm.getelementptr %15[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %c0_i64_8 = arith.constant 0 : i64
    llvm.store %c0_i64_8, %17 : i64, !llvm.ptr
    %18 = llvm.getelementptr %14[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %12, %18 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %19 = llvm.getelementptr %15[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %c1_i64_9 = arith.constant 1 : i64
    llvm.store %c1_i64_9, %19 : i64, !llvm.ptr
    %20 = call @__NS__MakeBuffer_f32(%14, %15, %13) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__Scatter(%8, %20) : (!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) -> ()
    %c0_i64_10 = arith.constant 0 : i64
    %21 = call @__NS__GetTensor_f32(%c0_i64_10, %20) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %22 = call @__NS__NSMemrefToMemref_f32(%21) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
    %23 = bufferization.to_memref %22 : memref<*xf32>
    %cast_11 = memref.cast %23 : memref<*xf32> to memref<1x128xf32, strided<[?, ?], offset: ?>>
    %24 = bufferization.to_tensor %cast_11 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %c1_i64_12 = arith.constant 1 : i64
    %25 = call @__NS__GetTensor_f32(%c1_i64_12, %20) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %26 = call @__NS__NSMemrefToMemref_f32(%25) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
    %27 = bufferization.to_memref %26 : memref<*xf32>
    %cast_13 = memref.cast %27 : memref<*xf32> to memref<1x128xf32, strided<[?, ?], offset: ?>>
    %28 = bufferization.to_tensor %cast_13 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    call @__NS__SetDevice(%c0_i64) : (i64) -> ()
    %29 = call @softmax_1_128_softmax_1_128_fused_kernel(%24) {device_id = 0 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %30 = bufferization.to_memref %29 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    memref.copy %30, %alloc_14 : memref<1x128xf32, strided<[?, ?], offset: ?>> to memref<1x128xf32>
    %cast_15 = memref.cast %alloc_14 : memref<1x128xf32> to memref<*xf32>
    %31 = bufferization.to_tensor %cast_15 : memref<*xf32>
    %c0_i64_16 = arith.constant 0 : i64
    %32 = call @__NS__MemrefToNSMemref_f32(%c0_i64_16, %31) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    call @__NS__SetDevice(%c1_i64) : (i64) -> ()
    %33 = call @softmax_1_128_softmax_1_128_fused_kernel(%28) {device_id = 1 : i64, device_kernel} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %34 = bufferization.to_memref %33 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    memref.copy %34, %alloc_17 : memref<1x128xf32, strided<[?, ?], offset: ?>> to memref<1x128xf32>
    %cast_18 = memref.cast %alloc_17 : memref<1x128xf32> to memref<*xf32>
    %35 = bufferization.to_tensor %cast_18 : memref<*xf32>
    %c1_i64_19 = arith.constant 1 : i64
    %36 = call @__NS__MemrefToNSMemref_f32(%c1_i64_19, %35) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %37 = llvm.mlir.constant(2 : i64) : i64
    %38 = llvm.alloca %37 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %39 = llvm.alloca %37 x i64 : (i64) -> !llvm.ptr
    %40 = llvm.getelementptr %38[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %32, %40 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %41 = llvm.getelementptr %39[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %c0_i64_20 = arith.constant 0 : i64
    llvm.store %c0_i64_20, %41 : i64, !llvm.ptr
    %42 = llvm.getelementptr %38[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %36, %42 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %43 = llvm.getelementptr %39[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %c1_i64_21 = arith.constant 1 : i64
    llvm.store %c1_i64_21, %43 : i64, !llvm.ptr
    %44 = call @__NS__MakeBuffer_f32(%38, %39, %37) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__SetDevice(%c0_i64) : (i64) -> ()
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<2x128xf32>
    %cast_23 = memref.cast %alloc_22 : memref<2x128xf32> to memref<*xf32>
    %45 = bufferization.to_tensor %cast_23 : memref<*xf32>
    %c0_i64_24 = arith.constant 0 : i64
    %46 = call @__NS__MemrefToNSMemref_f32(%c0_i64_24, %45) : (i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %47 = llvm.mlir.constant(1 : i64) : i64
    %48 = llvm.alloca %47 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %49 = llvm.alloca %47 x i64 : (i64) -> !llvm.ptr
    %50 = llvm.getelementptr %48[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %46, %50 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %51 = llvm.getelementptr %49[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %c0_i64_25 = arith.constant 0 : i64
    llvm.store %c0_i64_25, %51 : i64, !llvm.ptr
    %52 = call @__NS__MakeBuffer_f32(%48, %49, %47) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    call @__NS__Gather(%44, %52) : (!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) -> ()
    %c0_i64_26 = arith.constant 0 : i64
    %53 = call @__NS__GetTensor_f32(%c0_i64_26, %52) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %54 = call @__NS__NSMemrefToMemref_f32(%53) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
    %55 = bufferization.to_memref %54 : memref<*xf32>
    %cast_27 = memref.cast %55 : memref<*xf32> to memref<2x128xf32, strided<[?, ?], offset: ?>>
    %56 = bufferization.to_tensor %cast_27 : memref<2x128xf32, strided<[?, ?], offset: ?>>
    return %56 : tensor<2x128xf32>
  }
  func.func private @__NS__SetDevice(i64)
  func.func private @softmax_1_128_softmax_1_128_fused_kernel(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> attributes {device_kernel} {
    %0 = bufferization.to_memref %arg0 : memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.softmax dimension(1) ins(%0 : memref<1x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<1x128xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.softmax dimension(1) ins(%alloc : memref<1x128xf32>) outs(%alloc_0 : memref<1x128xf32>)
    %1 = bufferization.to_tensor %alloc_0 : memref<1x128xf32>
    return %1 : tensor<1x128xf32>
  }
  func.func private @__NS__MemrefToNSMemref_f32(i64, tensor<*xf32>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
  func.func private @__NS__MakeBuffer_f32(!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
  func.func private @__NS__Scatter(!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>)
  func.func private @__NS__GetTensor_f32(i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
  func.func private @__NS__NSMemrefToMemref_f32(!llvm.struct<(i64, struct<(i64, ptr)>)>) -> tensor<*xf32>
  func.func private @__NS__Gather(!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>)
}

// '/home/lfr/MLIR_Tutorial/build/third_party/llvm-project/llvm/bin/mlir-opt' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' -convert-linalg-to-loops -test-linalg-decompose-ops -func-bufferize  -canonicalize -finalizing-bufferize   -test-linalg-decompose-ops --mlir-print-ir-after-all -convert-linalg-to-affine-loops -buffer-results-to-out-params="hoist-static-allocs" --split-input-file -o '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test_out.mlir' &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'

// '/home/lfr/MLIR_Tutorial/build/15-lowing_to_llvm/src/Tools/NS-opt/NS-opt15' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' --convert-north-satr-to-func --mlir-print-ir-after-all -o '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test_out.mlir' --debug &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'

// '/home/lfr/MLIR_Tutorial/build/15-lowing_to_llvm/src/Tools/NS-opt/NS-opt15' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/to_llvm_pipeline.mlir' --north-star-basic-pipeline="DP_Nums=2" --mlir-print-ir-after-all   --debug &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'

// '/home/lfr/MLIR_Tutorial/build/15-lowing_to_llvm/src/Tools/NS-opt/NS-opt15' '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/test.mlir' --north-star-basic-pipeline="DP_Nums=2" --mlir-print-ir-after-all --debug &> '/home/lfr/MLIR_Tutorial/15-lowing_to_llvm/test/Pipeline/log.mlir'