#include <functional>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// It requires these headers, but does not include them.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

LogicalResult tpu_matmul_rule(tpu::MatmulOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto acc = op.getAcc();

  const VectorType lhs_ty = lhs.getType();
  const VectorType rhs_ty = rhs.getType();
  const VectorType acc_ty = acc.getType();

  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();
  auto acc_element_type = acc_ty.getElementType();

  auto extsi_sitofp = [&builder, &op](TypedValue<VectorType> element) {
    const VectorType ty = element.getType();
    auto shape = ty.getShape();
    CHECK(ty.getElementType().isInteger());
    TypedValue<VectorType> ext_ele;
    if (ty.getElementType().getIntOrFloatBitWidth() == 32) {
      ext_ele = element;
    } else {
      ext_ele = cast<TypedValue<VectorType>>(
          builder
              .create<arith::ExtSIOp>(
                  VectorType::get(shape, builder.getI32Type()), element)
              .getResult());
    }
    // TODO(mvoz): Go to bf16 when hardware supported, requires adding support
    // for 16 bitwidth in extsiop in infer/apply.
    auto ele_as_fp = builder.create<arith::SIToFPOp>(
        op.getLoc(), VectorType::get(shape, builder.getF32Type()), ext_ele);
    return ele_as_fp;
  };

  if (lhs_element_type != rhs_element_type) {
    if (lhs_element_type.isInteger() && rhs_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/int matmul.
      op->emitOpError("Mix int/int - NYI");
      return failure();
    }
    if (acc_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/float matmul with int acc.
      // Should be pretty straightforward.
      op->emitOpError("acc is int in mixed matmul - NYI");
      return failure();
    }
    if (lhs_element_type.isInteger()) {
      auto float_lhs = extsi_sitofp(lhs);
      op->setOperand(0, float_lhs);
    }
    if (rhs_element_type.isInteger()) {
      auto float_rhs = extsi_sitofp(rhs);
      op->setOperand(1, float_rhs);
    }
  }
  // TODO(mvoz): Add more invariants.
  if (acc_element_type.isInteger()) {
    CHECK(op.getLhs().getType().getElementType().isInteger());
    CHECK(op.getRhs().getType().getElementType().isInteger());
  } else {
    CHECK(!op.getLhs().getType().getElementType().isInteger());
    CHECK(!op.getRhs().getType().getElementType().isInteger());
  }
  return success();
};

// TODO(mvoz): If # of args before op grows > 2, we should consider creating
// a ctx object to pass around. A little overkill for now.
template <typename Op>
LogicalResult canonicalize_binops(int hardware_generation_, Op &op) {
  if (op.getNumOperands() != 2) {
    op.emitOpError("Invariant violated: Not a binary op");
    return failure();
  }
  auto lhs = op.getOperand(0);
  auto rhs = op.getOperand(1);
  auto lhs_ty = dyn_cast<VectorType>(lhs.getType());
  if (!lhs_ty) {
    op.emitOpError("Invariant violated: Not a vector");
    return failure();
  }
  auto rhs_ty = dyn_cast<VectorType>(rhs.getType());
  if (!rhs_ty) {
    op.emitOpError("Invariant violated: Not a vector");
    return failure();
  }
  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();
  if (lhs_element_type != rhs_element_type) {
    op.emitOpError("NYI - mixed type elementwise ops");
    return failure();
  }
  if ((!lhs_element_type.isF32() && !lhs_element_type.isBF16()) ||
      (!rhs_element_type.isF32() && !rhs_element_type.isBF16())) {
    op.emitOpError("NYI - non fp32/bf16 elementwise ops");
    return failure();
  }
  // bf16 is not supported in earlier hardware.
  if (hardware_generation_ <= 5) {
    // TODO(mvoz): Once we support mixed type elementwise ops, we need to
    // check a little more carefully here.
    if (lhs_element_type.isBF16() && rhs_element_type.isBF16()) {
      OpBuilder builder(op);
      auto target_f32_ty =
          VectorType::get(lhs_ty.getShape(), builder.getF32Type());
      auto target_bf16_ty =
          VectorType::get(rhs_ty.getShape(), builder.getBF16Type());
      auto target_f32_lhs =
          builder.create<arith::ExtFOp>(op.getLoc(), target_f32_ty, lhs)
              .getResult();
      auto target_f32_rhs =
          builder.create<arith::ExtFOp>(op.getLoc(), target_f32_ty, rhs)
              .getResult();
      auto op_in_f32 = builder.create<Op>(op.getLoc(), target_f32_ty,
                                          target_f32_lhs, target_f32_rhs);
      auto op_in_bf16 = builder.create<arith::TruncFOp>(
          op.getLoc(), target_bf16_ty, op_in_f32.getResult());
      op.replaceAllUsesWith(op_in_bf16.getResult());
      op.erase();
    }
  }
  return success();
}

LogicalResult canonicalize_matmul(Operation &op) {
  auto matmul_op = dyn_cast<tpu::MatmulOp>(op);
  if (!matmul_op) {
    op.emitOpError("Invariant violated: Not a matmul");
    return failure();
  }
  return tpu_matmul_rule(matmul_op);
};

LogicalResult canonicalize_contraction(Operation &op) {
  auto contraction_op = dyn_cast<vector::ContractionOp>(op);
  if (!contraction_op) {
    op.emitOpError("Invariant violated: Not a contraction");
    return failure();
  }
  // Rewrite the contraction as a matmul
  auto lhs = contraction_op.getLhs();
  auto rhs = contraction_op.getRhs();
  auto acc = contraction_op.getAcc();
  VectorType acc_ty;
  if (!(acc_ty = dyn_cast<VectorType>(acc.getType()))) {
    contraction_op->emitOpError("Not implemented: acc must be a vector");
    return failure();
  }

  if (contraction_op.getKind() != vector::CombiningKind::ADD) {
    contraction_op->emitOpError("Only ADD supported");
    return failure();
  }

  ImplicitLocOpBuilder builder(contraction_op->getLoc(),
                               contraction_op.getOperation());

  MLIRContext *const mlir_ctx = contraction_op->getContext();

  auto getMapAttr = [&](const unsigned first, const unsigned second) {
    return AffineMapAttr::get(AffineMap::get(
        3, 0,
        {getAffineDimExpr(first, mlir_ctx), getAffineDimExpr(second, mlir_ctx)},
        mlir_ctx));
  };

  const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
  const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
  const auto indexing_maps = contraction_op.getIndexingMaps();
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul or unsupported indexing_maps");
  }
  const bool transpose_rhs = indexing_maps == matmul_indexing_maps_transposed;

  const ArrayAttr matmul_iterator_types =
      builder.getArrayAttr({builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::reduction)});
  if (contraction_op->getAttr("iterator_types") != matmul_iterator_types) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      contraction_op->getAttrOfType<tpu::ContractPrecisionAttr>("precision");
  auto matmul_op = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false, transpose_rhs, precision_attr);
  contraction_op.replaceAllUsesWith(matmul_op.getResult());
  contraction_op.erase();
  auto result = tpu_matmul_rule(matmul_op);
  return result;
}

using canonicalize_rule_type = std::function<LogicalResult(Operation &op)>;

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction}};
  return *rules;
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer(int hardware_generation)
      : hardware_generation_(hardware_generation) {}

  int hardware_generation_;

  LogicalResult canonicalize(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return canonicalizeBlock(op.getBody().front());
  }

  LogicalResult canonicalizeBlock(Block &block) {
    // make_early_inc_range is utilized due to op mutation.
    for (Operation &any_op : make_early_inc_range(block)) {
      if (canonicalizeOp(any_op).failed()) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult canonicalizeOp(Operation &any_op) {
    // We must iterate over the op first, because canonicalization can cause
    // us to .erase() an op, and accessing getRegions on it after is not sound.
    // Invariant - top level ops with regions may never be invalidated.
    for (Region &region : any_op.getRegions()) {
      for (Block &block : region) {
        if (canonicalizeBlock(block).failed()) {
          return failure();
        }
      }
    }
    if (OpTrait::hasElementwiseMappableTraits(&any_op)) {
      // Is this the worst way of doing it .... ever? I would absolutely love
      // to be able to template on the op type somehow, but MLIR
      // has all this jank runtime casting stuff instead of proper
      // specialization, so I am unsure if we can do this.
      if (auto binop = dyn_cast<arith::MulFOp>(any_op)) {
        if (canonicalize_binops(hardware_generation_, binop).failed()) {
          return failure();
        }
      } else if (auto binop = dyn_cast<arith::DivFOp>(any_op)) {
        if (canonicalize_binops(hardware_generation_, binop).failed()) {
          return failure();
        }
      } else if (auto binop = dyn_cast<arith::AddFOp>(any_op)) {
        if (canonicalize_binops(hardware_generation_, binop).failed()) {
          return failure();
        }
      } else if (auto binop = dyn_cast<arith::SubFOp>(any_op)) {
        if (canonicalize_binops(hardware_generation_, binop).failed()) {
          return failure();
        }
      } else if (auto binop = dyn_cast<arith::MaximumFOp>(any_op)) {
        if (canonicalize_binops(hardware_generation_, binop).failed()) {
          return failure();
        }
      } else if (auto binop = dyn_cast<arith::MinimumFOp>(any_op)) {
        if (canonicalize_binops(hardware_generation_, binop).failed()) {
          return failure();
        }
      }
    }
    if (auto rule_it = rules().find(any_op.getName().getStringRef());
        rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      return rule(any_op);
    }
    return success();
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass(int hardware_generation)
      : hardware_generation_(hardware_generation) {}

  int hardware_generation_;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc(hardware_generation_);
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };
};

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation) {
  return std::make_unique<CanonicalizeMosaicPass>(hardware_generation);
}

}  // namespace mlir::tpu
