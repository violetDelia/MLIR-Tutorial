本教程的目标

本教程专为零基础或刚接触 MLIR 的开发者设计，聚焦 MLIR 的核心功能与工程实践，通过简洁的示例和规范的代码，手把手引导开发者掌握以下能力：

1. MLIR 开发环境搭建与工具链使用
2. 自定义方言（Dialect）与操作（Operation）的定义
3. 基础 IR 构建、转换与验证。

## 背景：

近年来，随着硬件架构的多样化（如 CPU、GPU、TPU、专用加速器）和计算需求的复杂化（如机器学习、高性能计算、边缘计算），传统的编译器技术逐渐面临瓶颈。单一固定的中间表示（IR，Intermediate Representation）难以高效应对不同领域的优化需求，而领域专用语言（DSL，Domain-Specific Language）的激增也加剧了工具链碎片化的问题。**MLIR（Multi-Level Intermediate Representation）** 正是在这一背景下诞生的编译器基础设施，它试图通过一种**可扩展、多层级、模块化**的设计哲学，为现代编译器技术的演进提供全新的可能性。

传统编译器（如 LLVM）的 IR 通常是静态且单一的，优化流程被固化在特定的抽象层次上。例如，LLVM IR 面向通用的低级机器指令，但在处理高阶语言特性（如 TensorFlow 的计算图或 PyTorch 的动态图）时，往往需要在早期“降低”（Lowering）过程中丢失大量语义信息，导致跨层级优化困难。MLIR 的核心洞察在于：**不同领域的计算问题需要不同抽象层次的中间表示，且这些表示应能共存、互操作与渐进式转换** 。通过允许开发者定义领域专用的 **方言（Dialect）** ，MLIR 使优化可以在最合适的抽象层级上进行，从而保留更多语义信息，解锁更深入的跨领域协同优化。

## 构建命令：

1. 拉包：`git clone https://github.com/violetDelia/MLIR-Tutorial.git`
2. 拉取第三方库：`git submodule update --init --recursive`
3. build：cd `./MLIR-Tutorial && cmake . build`
4. 编译：`cd build && ninja CH-2 [编译/测试命令]`

| CH                     | 目录                         | 编译命令   | 测试命令 | 视频链接 |
| ---------------------- | ---------------------------- | ---------- | -------- | -------- |
| 方言定义               | 2-define_dialect             | ninja CH-2 | NA       |          |
| Type定义               | 3-define_type                | ninja CH-3 | NA       |          |
| Attribute定义          | 4-define_attribute           | ninja CH-4 | NA       |          |
| Operation定义          | 5-define_operation           | ninja CH-5 | NA       |          |
| 接口定义               | 6-define_interface           |            |          |          |
| IR 结构                | 7-ir_struct                  |            |          |          |
| 定义Pass               | 8-define_pass                |            |          |          |
| 定义IR变换             | 9-rewrite_pattern            |            |          |          |
| debug以及mlir-opt工具  | 10-mlir_opt-and-debug        |            |          |          |
| Lit 测试套件           | 11-lit_for_test              |            |          |          |
| IR下降变换             | 12-operation_lowing_pass     |            |          |          |
| Pass 管理              | 13-pass_manager              |            |          |          |
| 规范化与常量折叠接口   | 14-fold_and_canonicalization |            |          |          |
| 下降到LLVM IR          | 15-lowing_to_llvm            |            |          |          |
| mlir-runner 工具执行IR | 16-mlir-runner               |            |          |          |
| IR端到端执行           | 17-execution                 |            |          |          |

## 备注：

1. 推荐使用g++ 编译器
