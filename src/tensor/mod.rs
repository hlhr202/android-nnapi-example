use crate::{context::ExecutionContext, primitives::NumericType};
use nnapi::{Burst, Compilation, Operand};
use nnapi_sys::OperandCode;
use std::sync::{Arc, Mutex};

pub struct Tensor<T: NumericType> {
    pub operand: Operand,
    pub data: Vec<T>,
    pub context: Arc<Mutex<ExecutionContext<T>>>,
}

// not a clean solution, should split context(similar to nn.module) and tensor
impl<T: NumericType> Tensor<T>
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

impl<T: NumericType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            operand: self.operand.clone(),
            data: self.data.clone(),
            // staging: None,
            context: self.context.clone(),
        }
    }
}

pub trait TensorForPrimitive<T: NumericType> {
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
