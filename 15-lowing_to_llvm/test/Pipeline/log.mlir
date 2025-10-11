initializing north_star
register north_star  Type
register north_star  Attr
register north_star  Op
module @NorthStar {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) {
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(8 : index) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(2 : index) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.constant(2 : i64) : i64
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.getelementptr %5[256] : (!llvm.ptr) -> !llvm.ptr, f32
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.add %14, %4 : i64
    %16 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %17 = llvm.call @__NS__Malloc(%15, %16) : (i64, !llvm.ptr) -> !llvm.ptr
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.sub %4, %6 : i64
    %20 = llvm.add %18, %19 : i64
    %21 = llvm.urem %20, %4  : i64
    %22 = llvm.sub %20, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %24 = llvm.insertvalue %17, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %23, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %3, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %8, %26[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %7, %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %7, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %6, %29[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mul %arg3, %6 : i64
    %32 = llvm.mul %31, %arg4 : i64
    %33 = llvm.getelementptr %5[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.mul %32, %34 : i64
    %36 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%23, %36, %35, %0, %37) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %38 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %30, %38 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %39 = llvm.call @__NS__MemrefToNSMemref_f32(%9, %8, %38) : (i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %40 = llvm.alloca %10 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %41 = llvm.alloca %10 x i64 : (i64) -> !llvm.ptr
    llvm.store %39, %40 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %42 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%17, %42) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.store %9, %41 : i64, !llvm.ptr
    %43 = llvm.call @__NS__MakeBuffer_f32(%40, %41, %10) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    llvm.call @__NS__SetDevice(%9) : (i64) -> ()
    %44 = llvm.getelementptr %5[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.add %45, %4 : i64
    %47 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %48 = llvm.call @__NS__Malloc(%46, %47) : (i64, !llvm.ptr) -> !llvm.ptr
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.add %49, %19 : i64
    %51 = llvm.urem %50, %4  : i64
    %52 = llvm.sub %50, %51 : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    %54 = llvm.insertvalue %48, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %53, %54[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %3, %55[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %6, %56[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %7, %57[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %7, %58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %6, %59[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %60, %61 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %62 = llvm.call @__NS__MemrefToNSMemref_f32(%9, %8, %61) : (i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.call @__NS__SetDevice(%10) : (i64) -> ()
    %63 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %64 = llvm.call @__NS__Malloc(%46, %63) : (i64, !llvm.ptr) -> !llvm.ptr
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.add %65, %19 : i64
    %67 = llvm.urem %66, %4  : i64
    %68 = llvm.sub %66, %67 : i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    %70 = llvm.insertvalue %64, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %69, %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %3, %71[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %6, %72[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.insertvalue %7, %73[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %7, %74[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %6, %75[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %76, %77 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %78 = llvm.call @__NS__MemrefToNSMemref_f32(%10, %8, %77) : (i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %79 = llvm.alloca %11 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %80 = llvm.alloca %11 x i64 : (i64) -> !llvm.ptr
    llvm.store %62, %79 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %81 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%48, %81) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.store %9, %80 : i64, !llvm.ptr
    %82 = llvm.getelementptr %79[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %78, %82 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %83 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%64, %83) : (!llvm.ptr, !llvm.ptr) -> ()
    %84 = llvm.getelementptr %80[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %10, %84 : i64, !llvm.ptr
    %85 = llvm.call @__NS__MakeBuffer_f32(%79, %80, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    llvm.call @__NS__Scatter(%43, %85) : (!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) -> ()
    %86 = llvm.call @__NS__GetTensor_f32(%9, %85) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %87 = llvm.call @__NS__NSMemrefToMemref_f32(%86) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> !llvm.struct<(i64, ptr)>
    %88 = llvm.mul %8, %1 : i64
    %89 = llvm.extractvalue %87[0] : !llvm.struct<(i64, ptr)> 
    %90 = llvm.mul %89, %8 : i64
    %91 = llvm.add %90, %6 : i64
    %92 = llvm.mul %91, %1 : i64
    %93 = llvm.add %88, %92 : i64
    %94 = llvm.alloca %93 x i8 : (i64) -> !llvm.ptr
    %95 = llvm.extractvalue %87[1] : !llvm.struct<(i64, ptr)> 
    %96 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%94, %95, %93, %0, %96) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %97 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%95, %97) : (!llvm.ptr, !llvm.ptr) -> ()
    %98 = llvm.load %94 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %99 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %100 = llvm.call @__NS__Malloc(%45, %99) : (i64, !llvm.ptr) -> !llvm.ptr
    %101 = llvm.insertvalue %100, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %100, %101[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.insertvalue %3, %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %6, %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %7, %104[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %7, %105[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %6, %106[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.intr.stacksave : !llvm.ptr
    %109 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %98, %109 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %110 = llvm.insertvalue %11, %2[0] : !llvm.struct<(i64, ptr)> 
    %111 = llvm.insertvalue %109, %110[1] : !llvm.struct<(i64, ptr)> 
    %112 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %107, %112 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %113 = llvm.insertvalue %112, %110[1] : !llvm.struct<(i64, ptr)> 
    %114 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %111, %114 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %115 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %113, %115 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %116 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__MemrefCopy(%34, %114, %115, %116) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %108 : !llvm.ptr
    %117 = llvm.call @__NS__GetTensor_f32(%10, %85) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %118 = llvm.call @__NS__NSMemrefToMemref_f32(%117) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> !llvm.struct<(i64, ptr)>
    %119 = llvm.extractvalue %118[0] : !llvm.struct<(i64, ptr)> 
    %120 = llvm.mul %119, %8 : i64
    %121 = llvm.add %120, %6 : i64
    %122 = llvm.mul %121, %1 : i64
    %123 = llvm.add %88, %122 : i64
    %124 = llvm.alloca %123 x i8 : (i64) -> !llvm.ptr
    %125 = llvm.extractvalue %118[1] : !llvm.struct<(i64, ptr)> 
    %126 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%124, %125, %123, %0, %126) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %127 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%125, %127) : (!llvm.ptr, !llvm.ptr) -> ()
    %128 = llvm.load %124 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %129 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %130 = llvm.call @__NS__Malloc(%45, %129) : (i64, !llvm.ptr) -> !llvm.ptr
    %131 = llvm.insertvalue %130, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %130, %131[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %3, %132[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %6, %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %7, %134[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.insertvalue %7, %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.insertvalue %6, %136[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.intr.stacksave : !llvm.ptr
    %139 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %128, %139 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %140 = llvm.insertvalue %139, %110[1] : !llvm.struct<(i64, ptr)> 
    %141 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %137, %141 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %142 = llvm.insertvalue %141, %110[1] : !llvm.struct<(i64, ptr)> 
    %143 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %140, %143 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %144 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %142, %144 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %145 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__MemrefCopy(%34, %143, %144, %145) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %138 : !llvm.ptr
    llvm.call @__NS__SetDevice(%9) : (i64) -> ()
    %146 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %147 = llvm.call @__NS__Malloc(%45, %146) : (i64, !llvm.ptr) -> !llvm.ptr
    llvm.call @softmax_1_128_softmax_1_128_fused_kernel(%100, %100, %3, %6, %7, %7, %6, %147, %147, %3, %6, %7, %7, %6) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    %148 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%100, %148) : (!llvm.ptr, !llvm.ptr) -> ()
    %149 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %150 = llvm.call @__NS__Malloc(%46, %149) : (i64, !llvm.ptr) -> !llvm.ptr
    %151 = llvm.ptrtoint %150 : !llvm.ptr to i64
    %152 = llvm.add %151, %19 : i64
    %153 = llvm.urem %152, %4  : i64
    %154 = llvm.sub %152, %153 : i64
    %155 = llvm.inttoptr %154 : i64 to !llvm.ptr
    %156 = llvm.insertvalue %150, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.insertvalue %155, %156[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.insertvalue %3, %157[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %6, %158[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.insertvalue %7, %159[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %7, %160[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %6, %161[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.mul %6, %6 : i64
    %164 = llvm.mul %163, %7 : i64
    %165 = llvm.mul %164, %34 : i64
    %166 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%155, %147, %165, %0, %166) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %167 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%147, %167) : (!llvm.ptr, !llvm.ptr) -> ()
    %168 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %162, %168 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %169 = llvm.call @__NS__MemrefToNSMemref_f32(%9, %8, %168) : (i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.call @__NS__SetDevice(%10) : (i64) -> ()
    %170 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %171 = llvm.call @__NS__Malloc(%45, %170) : (i64, !llvm.ptr) -> !llvm.ptr
    llvm.call @softmax_1_128_softmax_1_128_fused_kernel(%130, %130, %3, %6, %7, %7, %6, %171, %171, %3, %6, %7, %7, %6) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    %172 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%130, %172) : (!llvm.ptr, !llvm.ptr) -> ()
    %173 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %174 = llvm.call @__NS__Malloc(%46, %173) : (i64, !llvm.ptr) -> !llvm.ptr
    %175 = llvm.ptrtoint %174 : !llvm.ptr to i64
    %176 = llvm.add %175, %19 : i64
    %177 = llvm.urem %176, %4  : i64
    %178 = llvm.sub %176, %177 : i64
    %179 = llvm.inttoptr %178 : i64 to !llvm.ptr
    %180 = llvm.insertvalue %174, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %179, %180[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %3, %181[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.insertvalue %6, %182[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.insertvalue %7, %183[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.insertvalue %7, %184[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.insertvalue %6, %185[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.mul %6, %6 : i64
    %188 = llvm.mul %187, %7 : i64
    %189 = llvm.mul %188, %34 : i64
    %190 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%179, %171, %189, %0, %190) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %191 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%171, %191) : (!llvm.ptr, !llvm.ptr) -> ()
    %192 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %186, %192 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %193 = llvm.call @__NS__MemrefToNSMemref_f32(%10, %8, %192) : (i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %194 = llvm.alloca %11 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %195 = llvm.alloca %11 x i64 : (i64) -> !llvm.ptr
    llvm.store %169, %194 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %196 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%150, %196) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.store %9, %195 : i64, !llvm.ptr
    %197 = llvm.getelementptr %194[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, struct<(i64, ptr)>)>
    llvm.store %193, %197 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %198 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%174, %198) : (!llvm.ptr, !llvm.ptr) -> ()
    %199 = llvm.getelementptr %195[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %10, %199 : i64, !llvm.ptr
    %200 = llvm.call @__NS__MakeBuffer_f32(%194, %195, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    llvm.call @__NS__SetDevice(%9) : (i64) -> ()
    %201 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %202 = llvm.call @__NS__Malloc(%15, %201) : (i64, !llvm.ptr) -> !llvm.ptr
    %203 = llvm.ptrtoint %202 : !llvm.ptr to i64
    %204 = llvm.add %203, %19 : i64
    %205 = llvm.urem %204, %4  : i64
    %206 = llvm.sub %204, %205 : i64
    %207 = llvm.inttoptr %206 : i64 to !llvm.ptr
    %208 = llvm.insertvalue %202, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %209 = llvm.insertvalue %207, %208[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %210 = llvm.insertvalue %3, %209[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %211 = llvm.insertvalue %8, %210[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %212 = llvm.insertvalue %7, %211[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %213 = llvm.insertvalue %7, %212[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.insertvalue %6, %213[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %215 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %214, %215 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %216 = llvm.call @__NS__MemrefToNSMemref_f32(%9, %8, %215) : (i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %217 = llvm.alloca %10 x !llvm.struct<(i64, struct<(i64, ptr)>)> : (i64) -> !llvm.ptr
    %218 = llvm.alloca %10 x i64 : (i64) -> !llvm.ptr
    llvm.store %216, %217 : !llvm.struct<(i64, struct<(i64, ptr)>)>, !llvm.ptr
    %219 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%202, %219) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.store %9, %218 : i64, !llvm.ptr
    %220 = llvm.call @__NS__MakeBuffer_f32(%217, %218, %10) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)>
    llvm.call @__NS__Gather(%200, %220) : (!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) -> ()
    %221 = llvm.call @__NS__GetTensor_f32(%9, %220) : (i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)>
    %222 = llvm.call @__NS__NSMemrefToMemref_f32(%221) : (!llvm.struct<(i64, struct<(i64, ptr)>)>) -> !llvm.struct<(i64, ptr)>
    %223 = llvm.extractvalue %222[0] : !llvm.struct<(i64, ptr)> 
    %224 = llvm.mul %223, %8 : i64
    %225 = llvm.add %224, %6 : i64
    %226 = llvm.mul %225, %1 : i64
    %227 = llvm.add %88, %226 : i64
    %228 = llvm.alloca %227 x i8 : (i64) -> !llvm.ptr
    %229 = llvm.extractvalue %222[1] : !llvm.struct<(i64, ptr)> 
    %230 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%228, %229, %227, %0, %230) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %231 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%229, %231) : (!llvm.ptr, !llvm.ptr) -> ()
    %232 = llvm.load %228 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %233 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %234 = llvm.call @__NS__Malloc(%14, %233) : (i64, !llvm.ptr) -> !llvm.ptr
    %235 = llvm.insertvalue %234, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %236 = llvm.insertvalue %234, %235[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %237 = llvm.insertvalue %3, %236[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %238 = llvm.insertvalue %8, %237[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %239 = llvm.insertvalue %7, %238[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %240 = llvm.insertvalue %7, %239[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.insertvalue %6, %240[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.intr.stacksave : !llvm.ptr
    %243 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %232, %243 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %244 = llvm.insertvalue %243, %110[1] : !llvm.struct<(i64, ptr)> 
    %245 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %241, %245 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %246 = llvm.insertvalue %245, %110[1] : !llvm.struct<(i64, ptr)> 
    %247 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %244, %247 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %248 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %246, %248 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %249 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__MemrefCopy(%34, %247, %248, %249) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %242 : !llvm.ptr
    %250 = llvm.mul %8, %6 : i64
    %251 = llvm.mul %250, %7 : i64
    %252 = llvm.mul %251, %34 : i64
    %253 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %254 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%253, %234, %252, %0, %254) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %255 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%234, %255) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @__NS__SetDevice(i64) attributes {sym_visibility = "private"}
  llvm.func @softmax_1_128_softmax_1_128_fused_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) attributes {device_kernel, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %11 = llvm.call @__NS__Malloc(%9, %10) : (i64, !llvm.ptr) -> !llvm.ptr
    %12 = llvm.getelementptr %2[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %15 = llvm.call @__NS__Malloc(%13, %14) : (i64, !llvm.ptr) -> !llvm.ptr
    %16 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %17 = llvm.call @__NS__Malloc(%9, %16) : (i64, !llvm.ptr) -> !llvm.ptr
    %18 = llvm.add %13, %1 : i64
    %19 = "north_star.get_host_stream"() : () -> !llvm.ptr
    %20 = llvm.call @__NS__Malloc(%18, %19) : (i64, !llvm.ptr) -> !llvm.ptr
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.sub %1, %6 : i64
    %23 = llvm.add %21, %22 : i64
    %24 = llvm.urem %23, %1  : i64
    %25 = llvm.sub %23, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    llvm.store %5, %11 : f32, !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%27: i64):  // 2 preds: ^bb0, ^bb5
    %28 = llvm.icmp "slt" %27, %7 : i64
    llvm.cond_br %28, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%29: i64):  // 2 preds: ^bb2, ^bb4
    %30 = llvm.icmp "slt" %29, %7 : i64
    llvm.cond_br %30, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %31 = llvm.mul %3, %7 : i64
    %32 = llvm.add %31, %29 : i64
    %33 = llvm.getelementptr %arg1[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %34 = llvm.load %33 : !llvm.ptr -> f32
    %35 = llvm.load %11 : !llvm.ptr -> f32
    %36 = llvm.intr.maxnum(%34, %35)  : (f32, f32) -> f32
    llvm.store %36, %11 : f32, !llvm.ptr
    %37 = llvm.add %29, %6 : i64
    llvm.br ^bb3(%37 : i64)
  ^bb5:  // pred: ^bb3
    %38 = llvm.mul %3, %7 : i64
    %39 = llvm.add %38, %27 : i64
    %40 = llvm.getelementptr %arg1[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.load %11 : !llvm.ptr -> f32
    %43 = llvm.fsub %41, %42  : f32
    %44 = llvm.intr.exp(%43)  : (f32) -> f32
    %45 = llvm.getelementptr %15[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %44, %45 : f32, !llvm.ptr
    %46 = llvm.add %27, %6 : i64
    llvm.br ^bb1(%46 : i64)
  ^bb6:  // pred: ^bb1
    llvm.store %4, %11 : f32, !llvm.ptr
    llvm.store %5, %17 : f32, !llvm.ptr
    llvm.br ^bb7(%3 : i64)
  ^bb7(%47: i64):  // 2 preds: ^bb6, ^bb17
    %48 = llvm.icmp "slt" %47, %7 : i64
    llvm.cond_br %48, ^bb8, ^bb18
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%3 : i64)
  ^bb9(%49: i64):  // 2 preds: ^bb8, ^bb13
    %50 = llvm.icmp "slt" %49, %7 : i64
    llvm.cond_br %50, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%3 : i64)
  ^bb11(%51: i64):  // 2 preds: ^bb10, ^bb12
    %52 = llvm.icmp "slt" %51, %7 : i64
    llvm.cond_br %52, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %53 = llvm.mul %3, %7 : i64
    %54 = llvm.add %53, %51 : i64
    %55 = llvm.getelementptr %15[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 : !llvm.ptr -> f32
    %57 = llvm.load %11 : !llvm.ptr -> f32
    %58 = llvm.fadd %56, %57  : f32
    llvm.store %58, %11 : f32, !llvm.ptr
    %59 = llvm.add %51, %6 : i64
    llvm.br ^bb11(%59 : i64)
  ^bb13:  // pred: ^bb11
    %60 = llvm.load %11 : !llvm.ptr -> f32
    %61 = llvm.mul %3, %7 : i64
    %62 = llvm.add %61, %49 : i64
    %63 = llvm.getelementptr %15[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.load %63 : !llvm.ptr -> f32
    %65 = llvm.fdiv %64, %60  : f32
    llvm.store %65, %63 : f32, !llvm.ptr
    %66 = llvm.add %49, %6 : i64
    llvm.br ^bb9(%66 : i64)
  ^bb14:  // pred: ^bb9
    llvm.br ^bb15(%3 : i64)
  ^bb15(%67: i64):  // 2 preds: ^bb14, ^bb16
    %68 = llvm.icmp "slt" %67, %7 : i64
    llvm.cond_br %68, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %69 = llvm.mul %3, %7 : i64
    %70 = llvm.add %69, %67 : i64
    %71 = llvm.getelementptr %15[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.load %71 : !llvm.ptr -> f32
    %73 = llvm.load %17 : !llvm.ptr -> f32
    %74 = llvm.intr.maxnum(%72, %73)  : (f32, f32) -> f32
    llvm.store %74, %17 : f32, !llvm.ptr
    %75 = llvm.add %67, %6 : i64
    llvm.br ^bb15(%75 : i64)
  ^bb17:  // pred: ^bb15
    %76 = llvm.mul %3, %7 : i64
    %77 = llvm.add %76, %47 : i64
    %78 = llvm.getelementptr %15[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %79 = llvm.load %78 : !llvm.ptr -> f32
    %80 = llvm.load %17 : !llvm.ptr -> f32
    %81 = llvm.fsub %79, %80  : f32
    %82 = llvm.intr.exp(%81)  : (f32) -> f32
    %83 = llvm.getelementptr %26[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %82, %83 : f32, !llvm.ptr
    %84 = llvm.add %47, %6 : i64
    llvm.br ^bb7(%84 : i64)
  ^bb18:  // pred: ^bb7
    llvm.store %4, %17 : f32, !llvm.ptr
    llvm.br ^bb19(%3 : i64)
  ^bb19(%85: i64):  // 2 preds: ^bb18, ^bb23
    %86 = llvm.icmp "slt" %85, %7 : i64
    llvm.cond_br %86, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%3 : i64)
  ^bb21(%87: i64):  // 2 preds: ^bb20, ^bb22
    %88 = llvm.icmp "slt" %87, %7 : i64
    llvm.cond_br %88, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %89 = llvm.mul %3, %7 : i64
    %90 = llvm.add %89, %87 : i64
    %91 = llvm.getelementptr %26[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.load %91 : !llvm.ptr -> f32
    %93 = llvm.load %17 : !llvm.ptr -> f32
    %94 = llvm.fadd %92, %93  : f32
    llvm.store %94, %17 : f32, !llvm.ptr
    %95 = llvm.add %87, %6 : i64
    llvm.br ^bb21(%95 : i64)
  ^bb23:  // pred: ^bb21
    %96 = llvm.load %17 : !llvm.ptr -> f32
    %97 = llvm.mul %3, %7 : i64
    %98 = llvm.add %97, %85 : i64
    %99 = llvm.getelementptr %26[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.load %99 : !llvm.ptr -> f32
    %101 = llvm.fdiv %100, %96  : f32
    llvm.store %101, %99 : f32, !llvm.ptr
    %102 = llvm.add %85, %6 : i64
    llvm.br ^bb19(%102 : i64)
  ^bb24:  // pred: ^bb19
    %103 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%17, %103) : (!llvm.ptr, !llvm.ptr) -> ()
    %104 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%15, %104) : (!llvm.ptr, !llvm.ptr) -> ()
    %105 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%11, %105) : (!llvm.ptr, !llvm.ptr) -> ()
    %106 = llvm.mul %6, %6 : i64
    %107 = llvm.mul %106, %7 : i64
    %108 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.mul %107, %109 : i64
    %111 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Memcpy(%111, %26, %110, %0, %112) : (!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr) -> ()
    %113 = "north_star.get_host_stream"() : () -> !llvm.ptr
    llvm.call @__NS__Free(%20, %113) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @__NS__MemrefToNSMemref_f32(i64, i64, !llvm.ptr) -> !llvm.struct<(i64, struct<(i64, ptr)>)> attributes {sym_visibility = "private"}
  llvm.func @__NS__MakeBuffer_f32(!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(i64, ptr, ptr, ptr)> attributes {sym_visibility = "private"}
  llvm.func @__NS__Scatter(!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) attributes {sym_visibility = "private"}
  llvm.func @__NS__GetTensor_f32(i64, !llvm.struct<(i64, ptr, ptr, ptr)>) -> !llvm.struct<(i64, struct<(i64, ptr)>)> attributes {sym_visibility = "private"}
  llvm.func @__NS__NSMemrefToMemref_f32(!llvm.struct<(i64, struct<(i64, ptr)>)>) -> !llvm.struct<(i64, ptr)> attributes {sym_visibility = "private"}
  llvm.func @__NS__Gather(!llvm.struct<(i64, ptr, ptr, ptr)>, !llvm.struct<(i64, ptr, ptr, ptr)>) attributes {sym_visibility = "private"}
  llvm.func @__NS__Free(!llvm.ptr, !llvm.ptr)
  llvm.func @__NS__Memcpy(!llvm.ptr, !llvm.ptr, i64, i1, !llvm.ptr)
  llvm.func @__NS__Malloc(i64, !llvm.ptr) -> !llvm.ptr
  llvm.func @__NS__MemrefCopy(i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
}

destroying north_star
