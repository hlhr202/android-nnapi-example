use crate::{context::ExecutionContext, tensor::Tensor};
use nnapi::Device;

pub fn test_nn() -> nnapi::Result<()> {
    let devices = Device::get_devices()?;
    println!("devices: {:?}", devices); // [Device: nnapi-reference]

    let context = ExecutionContext::new(devices.clone());

    let tensor1 = Tensor::new(vec![1., 2., 3.], vec![3], context.clone());
    let tensor2 = Tensor::new(vec![4., 4., 7.], vec![3], context.clone());
    let tensor3 = tensor1.clone();

    let mut output = tensor1 + tensor2 + tensor3;

    println!("output: {:?}", output.get_data()); // [6.0, 8.0, 13.0]

    let context = ExecutionContext::new(devices.clone());

    let tensor2x3 = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], context.clone());
    let tensor3x2 = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2], context.clone());
    let mut output = tensor2x3.matmul(tensor3x2);
    println!("output2: {:?}", output.get_data()); // [22.0, 28.0, 49.0, 64.0]

    Ok(())
}

// pub fn _test_multiple_inout() -> nnapi::Result<()> {
//     let op_input = Operand::tensor(OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32, vec![3], 0., 0);
//     let op1 = Operand::tensor(OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32, vec![3], 0., 0);
//     let op_inter = Operand::tensor(OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32, vec![3], 0., 0);
//     let op2 = Operand::tensor(OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32, vec![3], 0., 0);
//     let op_output = Operand::tensor(OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32, vec![3], 0., 0);

//     let activation1 = Operand::activation();
//     let activation2 = Operand::activation();

//     let mut model = Model::from_operands([
//         op_input,    // 0
//         op1,         // 1
//         activation1, // 2
//         op_inter,    // 3
//         op2,         // 4
//         activation2, // 5
//         op_output,   // 6
//     ])?;

//     model.set_activation_operand_value(2)?;
//     model.set_activation_operand_value(5)?;
//     model.set_operand_value(1, &[0f32, 1., 2.])?;
//     model.set_operand_value(4, &[3f32, 4., 5.])?;

//     model.add_operation(OperationCode::ANEURALNETWORKS_ADD, &[0, 1, 2], &[3])?;
//     model.add_operation(OperationCode::ANEURALNETWORKS_ADD, &[3, 4, 5], &[6])?;
//     model.identify_inputs_and_outputs(&[0], &[6])?;
//     model.finish()?;

//     let mut compilation = model.compile()?;
//     compilation.finish()?;

//     let mut execution = compilation.create_execution()?;
//     execution.set_input(0, &[1f32, 2., 3.])?;

//     let mut output = [0f32; 3];
//     execution.set_output(0, &mut output)?;
//     let mut event = execution.compute()?;
//     event.wait()?;

//     println!("output: {:?}", output); // [4.0, 7.0, 10.0]

//     Ok(())
// }
