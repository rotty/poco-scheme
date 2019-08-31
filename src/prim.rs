use std::io::{self, Write};

use crate::value::{PrimOp, Value};

fn invalid_argument(arg: Value, expected: &str) -> Value {
    make_error!("invalid argument, expected {}", expected; arg)
}

fn too_few_arguments(procedure: &str, args: &[Value]) -> Value {
    make_error!("too few arguments to `{}'", procedure; Value::list(args.iter().cloned()))
}

fn wrong_number_of_arguments(procedure: &str, expected: usize, args: &[Value]) -> Value {
    make_error!(
        "wrong number of arguments to `{}': expected {}, got {}",
        procedure,
        expected,
        args.len();
        Value::list(args.iter().cloned())
    )
}

fn io_error(e: io::Error) -> Value {
    make_error!("I/O error: {}", e)
}

fn arithmetic_overflow(operation: &str, arg1: isize, arg2: isize) -> Value {
    make_error!(
        "arithmetic overflow in {} of {} and {}",
        operation,
        arg1,
        arg2
    )
}

fn fixnum_arg(value: &Value) -> Result<isize, Value> {
    value
        .as_fixnum()
        .ok_or_else(|| invalid_argument(value.clone(), "fixnum"))
}

pub fn plus(args: &[Value]) -> Value {
    if let Some((first, rest)) = args.split_first() {
        let mut sum = try_result!(fixnum_arg(first));
        for elt in rest {
            let n = try_result!(fixnum_arg(elt));
            sum = try_result!(sum
                .checked_add(n)
                .ok_or_else(|| arithmetic_overflow("addition", sum, n)));
        }
        Value::Fixnum(sum)
    } else {
        Value::number(0isize)
    }
}

pub fn minus(args: &[Value]) -> Value {
    if let Some((first, rest)) = args.split_first() {
        let mut sum = try_result!(fixnum_arg(first));
        for elt in rest {
            let n = try_result!(fixnum_arg(elt));
            sum = try_result!(sum
                .checked_sub(n)
                .ok_or_else(|| arithmetic_overflow("addition", sum, n)));
        }
        Value::Fixnum(sum)
    } else {
        too_few_arguments("-", args)
    }
}

pub fn times(args: &[Value]) -> Value {
    if let Some((first, rest)) = args.split_first() {
        let mut product = try_result!(fixnum_arg(first));
        for elt in rest {
            let n = try_result!(fixnum_arg(elt));
            product = try_result!(product.checked_mul(n).ok_or_else(|| arithmetic_overflow(
                "multiplication",
                product,
                n
            )));
        }
        Value::Fixnum(product)
    } else {
        Value::Fixnum(1)
    }
}

fn num_cmp<F>(args: &[Value], cmp: F) -> Value
where
    F: Fn(&isize, &isize) -> bool,
{
    // TODO: this does the fixnum conversion more often than necessary, since
    // the windows overlap.
    for w in args.windows(2) {
        let n1 = try_result!(fixnum_arg(&w[0]));
        let n2 = try_result!(fixnum_arg(&w[1]));
        if !cmp(&n1, &n2) {
            return Value::from(false);
        }
    }
    Value::from(true)
}

pub fn eq(args: &[Value]) -> Value {
    num_cmp(args, isize::ge)
}

pub fn lt(args: &[Value]) -> Value {
    num_cmp(args, isize::lt)
}

pub fn le(args: &[Value]) -> Value {
    num_cmp(args, isize::le)
}

pub fn gt(args: &[Value]) -> Value {
    num_cmp(args, isize::gt)
}

pub fn ge(args: &[Value]) -> Value {
    num_cmp(args, isize::ge)
}

pub fn modulo(args: &[Value]) -> Value {
    if args.len() != 2 {
        return wrong_number_of_arguments("mod", 2, args);
    }
    let n1 = try_result!(fixnum_arg(&args[0]));
    let n2 = try_result!(fixnum_arg(&args[1]));
    Value::Fixnum(n1 % n2)
}

pub fn sqrt(args: &[Value]) -> Value {
    if args.len() != 1 {
        return wrong_number_of_arguments("sqrt", 1, args);
    }
    let n = try_result!(fixnum_arg(&args[0]));
    Value::Fixnum((n as f64).sqrt() as isize)
}

pub fn display(args: &[Value]) -> Value {
    if args.len() != 1 {
        // TODO: support ports
        return wrong_number_of_arguments("display", 1, args);
    }
    // TODO: we use the `Display` trait of `Value` here, which currently
    // uses `write` notation, not `display` notation.
    try_result!(write!(io::stdout(), "{}", args[0]).map_err(io_error));
    Value::Unspecified
}

pub fn newline(args: &[Value]) -> Value {
    if !args.is_empty() {
        // TODO: support ports
        return wrong_number_of_arguments("newline", 0, args);
    }
    try_result!(writeln!(io::stdout()).map_err(io_error));
    Value::Unspecified
}

macro_rules! prim_op {
    ($name:tt, $func:expr) => {{
        static OP: PrimOp = PrimOp {
            name: $name,
            func: $func,
        };
        ($name, Value::PrimOp(&OP))
    }};
}

pub fn make_ops() -> Vec<(&'static str, Value)> {
    vec![
        prim_op!("+", plus),
        prim_op!("-", minus),
        prim_op!("*", times),
        prim_op!("<", lt),
        prim_op!("<=", le),
        prim_op!(">", gt),
        prim_op!(">=", ge),
        prim_op!("=", eq),
        prim_op!("modulo", modulo),
        prim_op!("sqrt", sqrt),
        prim_op!("display", display),
        prim_op!("newline", newline),
    ]
}
