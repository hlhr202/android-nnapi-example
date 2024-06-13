use crate::{context::ExecutionContext, primitives::DataPrimitive};
use nnapi::{Burst, Compilation, Operand};
use nnapi_sys::{OperandCode, OperationCode};
use std::{
    fmt::Debug,
    ops::Add,
    sync::{Arc, Mutex},
};

pub struct Tensor<T: Clone + DataPrimitive + Debug + Send + Sync> {
    operand: Operand,
    data: Vec<T>,
    context: Arc<Mutex<ExecutionContext<T>>>,
}

// not a clean solution, should split context(similar to nn.module) and tensor
impl<T: Clone + DataPrimitive + Debug + Send + Sync> Tensor<T>
where
    Tensor<T>: TensorForPrimitive<T>,
{
    pub fn new(
        data: Vec<T>,
        dimensions: Vec<u32>,
        context: Arc<Mutex<ExecutionContext<T>>>,
    ) -> Self {
        TensorForPrimitive::new(data, dimensions, context)
    }

    fn add_ops(self, other: Self) -> Self {
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

    pub fn matmul(self, other: Self) -> Self {
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

    pub fn compile_compute_pipeline(&self) -> Compilation {
        {
            let context_ref = self.context.clone();
            let mut context = context_ref.lock().unwrap();
            let count = context.count;
            context
                .model
                .identify_inputs_and_outputs(&[0], &[count as u32])
                .unwrap();
            context.model.finish().unwrap();

            let devices = context.devices.clone();
            let mut compilation = context.model.compile_for_devices(devices).unwrap();
            compilation.finish().unwrap();

            compilation
        }
    }

    #[allow(dead_code)]
    pub fn compute_as_output(&mut self, compilation: &mut Compilation) {
        {
            let context_ref = self.context.clone();
            let context = context_ref.lock().unwrap();

            let mut execution = compilation.create_execution().unwrap();
            execution
                .set_input(0, &context.input.as_ref().unwrap().data)
                .unwrap();
            execution.set_output(0, &mut self.data).unwrap();

            let mut event = execution.compute().unwrap();
            event.wait().unwrap();
        }
    }

    pub fn burst_compute_as_output(&mut self, compilation: &mut Compilation, burst: &mut Burst) {
        {
            let context_ref = self.context.clone();
            let context = context_ref.lock().unwrap();

            let mut execution = compilation.create_execution().unwrap();
            execution
                .set_input(0, &context.input.as_ref().unwrap().data)
                .unwrap();
            execution.set_output(0, &mut self.data).unwrap();

            execution.burst_compute(burst).unwrap();
        }
    }

    pub fn get_data(&mut self) -> &Vec<T> {
        let mut compilation = self.compile_compute_pipeline();
        let mut burst = Burst::new(&mut compilation).unwrap();
        self.burst_compute_as_output(&mut compilation, &mut burst);
        &self.data
    }
}

impl<T: Clone + DataPrimitive + Debug + Send + Sync> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            operand: self.operand.clone(),
            data: self.data.clone(),
            // staging: None,
            context: self.context.clone(),
        }
    }
}

impl Add for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add_ops(self, rhs)
    }
}

pub trait TensorForPrimitive<T: Clone + DataPrimitive + Debug + Send + Sync> {
    fn new(data: Vec<T>, dimensions: Vec<u32>, context: Arc<Mutex<ExecutionContext<T>>>) -> Self;
}

impl TensorForPrimitive<f32> for Tensor<f32> {
    fn new(
        data: Vec<f32>,
        dimensions: Vec<u32>,
        context: Arc<Mutex<ExecutionContext<f32>>>,
    ) -> Self {
        let operand = Operand::tensor(
            OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32,
            dimensions,
            0.,
            0,
        );

        Self {
            operand,
            data,
            context,
        }
    }
}

impl TensorForPrimitive<i32> for Tensor<i32> {
    fn new(
        data: Vec<i32>,
        dimensions: Vec<u32>,
        context: Arc<Mutex<ExecutionContext<i32>>>,
    ) -> Self {
        let operand = Operand::tensor(OperandCode::ANEURALNETWORKS_TENSOR_INT32, dimensions, 0., 0);

        Self {
            operand,
            data,
            context,
        }
    }
}
