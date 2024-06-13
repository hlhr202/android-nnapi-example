pub trait DataPrimitive {
    fn zero() -> Self;
}

impl DataPrimitive for f32 {
    fn zero() -> Self {
        0.
    }
}

impl DataPrimitive for i32 {
    fn zero() -> Self {
        0
    }
}