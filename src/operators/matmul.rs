use crate::{
    primitives::NumericType,
    tensor::{Tensor, TensorForPrimitive},
};
use nnapi::Operand;
use nnapi_sys::OperationCode;

pub trait MatMulOp {
    fn matmul(self, other: Self) -> Self;
}

impl<T: NumericType> MatMulOp for Tensor<T>
where
    Tensor<T>: TensorForPrimitive<T>,
{
    fn matmul(self, other: Self) -> Self {
        assert_eq!(
            self.operand.dimensions.last(),
            other.operand.dimensions.first()
        );

        // compute by out shape
        // eg shape1: [2,3,4] shape2: [4,5,6] -> output shape: [2,3,5,6]
        let new_shape = self.operand.dimensions[..self.operand.dimensions.len() - 1]
            .iter()
            .chain(other.operand.dimensions[1..].iter())
            .cloned()
            .collect::<Vec<_>>();

        let new_len = new_shape.iter().product::<u32>() as usize;
        let output = Tensor::<T>::new(vec![T::zero(); new_len], new_shape, self.context.clone());

        let context_ref = self.context.clone();
        let mut context = context_ref.lock().unwrap();

        let bool_scalar1 = Operand::bool_scalar();
        let bool_scalar2 = Operand::bool_scalar();
        let transpose = [0u8];

        let mut operands = vec![
            &other.operand,
            &bool_scalar1,
            &bool_scalar2,
            &output.operand,
        ];

        let is_head = context.input.is_none();
        if is_head {
            operands.insert(0, &self.operand);
        }

        for operand in operands.iter() {
            context.model.add_operand(operand).unwrap();
        }

        if is_head {
            context.input = Some(self);
        } else {
            context.operators.push(self);
        }

        let count = context.count;
        let op1_idx = count as u32;
        let op2_idx = count as u32 + 1;
        let transpose1_idx = count as u32 + 2;
        let transpose2_idx = count as u32 + 3;
        let output_idx = count as u32 + 4;

        context
            .model
            .set_operand_value(transpose1_idx as i32, &transpose)
            .unwrap();
        context
            .model
            .set_operand_value(transpose2_idx as i32, &transpose)
            .unwrap();
        context
            .model
            .set_operand_value(op2_idx as i32, &other.data)
            .unwrap();

        let matmul_inputs_idx = [op1_idx, op2_idx, transpose1_idx, transpose2_idx];
        let matmul_output_idx = [output_idx];

        context
            .model
            .add_operation(
                OperationCode::ANEURALNETWORKS_BATCH_MATMUL,
                &matmul_inputs_idx,
                &matmul_output_idx,
            )
            .unwrap();

        context.operators.push(other);

        context.count += 4;

        output
    }
}
