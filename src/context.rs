use crate::{primitives::NumericType, tensor::Tensor};
use nnapi::{Device, Model};
use std::sync::{Arc, Mutex};

pub struct ExecutionContext<T: NumericType> {
    pub model: Model,
    pub input: Option<Tensor<T>>,
    pub operators: Vec<Tensor<T>>,
    pub count: usize,
    pub devices: Vec<Device>,
}

impl<T: NumericType> ExecutionContext<T> {
    pub fn new(devices: Vec<Device>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            model: Model::new().unwrap(),
            input: None,
            operators: vec![],
            count: 0,
            devices,
        }))
    }
}
