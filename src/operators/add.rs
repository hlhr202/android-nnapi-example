use crate::{
    primitives::NumericType,
    tensor::{Tensor, TensorForPrimitive},
};
use nnapi::Operand;
use nnapi_sys::OperationCode;
use std::ops::Add;

pub trait AddOp {
    fn add_(self, other: Self) -> Self;
}

impl<T: NumericType> AddOp for Tensor<T>
where
    Tensor<T>: TensorForPrimitive<T>,
{
    fn add_(self, other: Self) -> Self {
        let output = Tensor::<T>::new(
            vec![T::zero(); self.data.len()],
            self.operand.dimensions.clone(),
            self.context.clone(),
        );

        let context_ref = self.context.clone();
        let mut context = context_ref.lock().unwrap();

        let activation = Operand::activation();
        let mut operands = vec![&other.operand, &activation, &output.operand];

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
        let activation_idx = count as u32 + 2;
        let output_idx = count as u32 + 3;
        let add_inputs_idx = [op1_idx, op2_idx, activation_idx];
        let add_output_idx = [output_idx];
        context
            .model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        context
            .model
            .set_operand_value(op2_idx as i32, &other.data)
            .unwrap();
        context
            .model
            .add_operation(
                OperationCode::ANEURALNETWORKS_ADD,
                &add_inputs_idx,
                &add_output_idx,
            )
            .unwrap();

        context.operators.push(other);

        context.count += 3;

        output
    }
}

impl Add for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add_(self, rhs)
    }
}

impl Add for Tensor<i32> {
    type Output = Tensor<i32>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add_(self, rhs)
    }
}
