use std::io::{self, Write};

use crate::{OpResult, Value};

fn invalid_argument(arg: &Value, expected: &str) -> Value {
    make_error!("invalid argument: {}, expected {}", arg, expected)
}

fn too_few_arguments(procedure: &str) -> Value {
    make_error!("too few arguments to `{}'", procedure)
}

fn wrong_number_of_arguments(procedure: &str, expected: usize, args: &[Value]) -> Value {
    make_error!(
        "wrong number of arguments to `{}': expected {}, got {}",
        procedure,
        expected,
        args.len()
    )
}

fn io_error(e: io::Error) -> Value {
    make_error!("I/O error: {}", e)
}

fn arithmetic_overflow(operation: &str, arg1: i64, arg2: i64) -> Value {
    make_error!(
        "arithmetic overflow in {} of {} and {}",
        operation,
        arg1,
        arg2
    )
}

pub fn plus(args: &[Value]) -> OpResult {
    if let Some((first, rest)) = args.split_first() {
        let mut sum = first
            .as_i64()
            .ok_or_else(|| invalid_argument(first, "number"))?
            .clone();
        for elt in rest {
            let n = elt
                .as_i64()
                .ok_or_else(|| invalid_argument(elt, "number"))?;
            sum = sum
                .checked_add(n)
                .ok_or_else(|| arithmetic_overflow("addition", sum, n))?;
        }
        Ok(Value::Fixnum(sum).into())
    } else {
        Ok(Value::number(0).into())
    }
}

pub fn minus(args: &[Value]) -> OpResult {
    if let Some((first, rest)) = args.split_first() {
        let mut sum = first
            .as_i64()
            .ok_or_else(|| invalid_argument(first, "number"))?
            .clone();
        for elt in rest {
            let n = elt
                .as_i64()
                .ok_or_else(|| invalid_argument(elt, "number"))?;
            sum = sum
                .checked_sub(n)
                .ok_or_else(|| arithmetic_overflow("addition", sum, n))?;
        }
        Ok(Value::Fixnum(sum).into())
    } else {
        Err(too_few_arguments("-"))
    }
}

pub fn times(args: &[Value]) -> OpResult {
    if let Some((first, rest)) = args.split_first() {
        let mut product = first
            .as_i64()
            .ok_or_else(|| invalid_argument(first, "number"))?
            .clone();
        for elt in rest {
            let n = elt
                .as_i64()
                .ok_or_else(|| invalid_argument(elt, "number"))?;
            product = product
                .checked_mul(n)
                .ok_or_else(|| arithmetic_overflow("multiplication", product, n))?;
        }
        Ok(Value::Fixnum(product).into())
    } else {
        Ok(Value::number(1).into())
    }
}

fn num_cmp<F>(args: &[Value], cmp: F) -> OpResult
where
    F: Fn(&i64, &i64) -> bool,
{
    for w in args.windows(2) {
        let n1 = w[0]
            .as_i64()
            .ok_or_else(|| invalid_argument(&w[0], "number"))?;
        let n2 = w[1]
            .as_i64()
            .ok_or_else(|| invalid_argument(&w[1], "number"))?;
        if !cmp(&n1, &n2) {
            return Ok(Value::from(false));
        }
    }
    Ok(Value::from(true))
}

pub fn eq(args: &[Value]) -> OpResult {
    num_cmp(args, i64::ge)
}

pub fn lt(args: &[Value]) -> OpResult {
    num_cmp(args, i64::lt)
}

pub fn le(args: &[Value]) -> OpResult {
    num_cmp(args, i64::le)
}

pub fn gt(args: &[Value]) -> OpResult {
    num_cmp(args, i64::gt)
}

pub fn ge(args: &[Value]) -> OpResult {
    num_cmp(args, i64::ge)
}

pub fn display(args: &[Value]) -> OpResult {
    if args.len() != 1 {
        // TODO: support ports
        return Err(wrong_number_of_arguments("display", 1, args));
    }
    // TODO: we use the `Display` trait of `Value` here, which currently
    // uses `write` notation, not `display` notation.
    write!(io::stdout(), "{}", args[0]).map_err(io_error)?;
    Ok(Value::Null)
}

pub fn newline(args: &[Value]) -> OpResult {
    if args.len() != 0 {
        // TODO: support ports
        return Err(wrong_number_of_arguments("newline", 0, args));
    }
    write!(io::stdout(), "\n").map_err(io_error)?;
    Ok(Value::Null)
}
