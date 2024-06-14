use std::fmt::Debug;

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.
    }
}

impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}

pub trait NumericType: Zero + Clone + Copy + Debug + Send + Sync {}

impl NumericType for f32 {}

impl NumericType for i32 {}
