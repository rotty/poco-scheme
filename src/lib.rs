macro_rules! make_error {
    ($fmt:literal, $($fmtargs:expr),*) => {
        $crate::Value::Exception(Box::new($crate::value::Exception {
            message: format!($fmt, $($fmtargs),*),
            irritants: Vec::new(),
        }))
    };
    ($fmt:literal; $($irritants:expr),*) => {
        $crate::Value::Exception(Box::new($crate::value::Exception {
            message: $fmt.into(),
            irritants: vec![$($irritants),*],
        }))
    };
    ($fmt:literal, $($fmtargs:expr),*; $($irritants:expr),*) => {
        $crate::Value::Exception(Box::new($crate::value::Exception {
            message: format!($fmt, $($fmtargs),*),
            irritants: vec![$($irritants),*],
        }))
    };
}

macro_rules! try_value {
    ($expr:expr) => {{
        let v = $expr;
        if let Value::Exception(_) = v {
            return v.into();
        } else {
            v
        }
    }};
}

macro_rules! try_result {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => return e.into(),
        }
    };
}

mod ast;
mod prim;
mod util;
mod value;
mod vm;

pub use value::{PrimOp, Value};
pub use vm::{EvalError, Vm};
