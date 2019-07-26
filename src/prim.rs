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

fn arithmetic_overflow(operation: &str, arg1: isize, arg2: isize) -> Value {
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
            .as_fixnum()
            .ok_or_else(|| invalid_argument(first, "number"))?;
        for elt in rest {
            let n = elt
                .as_fixnum()
                .ok_or_else(|| invalid_argument(elt, "number"))?;
            sum = sum
                .checked_add(n)
                .ok_or_else(|| arithmetic_overflow("addition", sum, n))?;
        }
        Ok(Value::Fixnum(sum))
    } else {
        Ok(Value::number(0isize))
    }
}

pub fn minus(args: &[Value]) -> OpResult {
    if let Some((first, rest)) = args.split_first() {
        let mut sum = first
            .as_fixnum()
            .ok_or_else(|| invalid_argument(first, "number"))?;
        for elt in rest {
            let n = elt
                .as_fixnum()
                .ok_or_else(|| invalid_argument(elt, "number"))?;
            sum = sum
                .checked_sub(n)
                .ok_or_else(|| arithmetic_overflow("addition", sum, n))?;
        }
        Ok(Value::Fixnum(sum))
    } else {
        Err(too_few_arguments("-"))
    }
}

pub fn times(args: &[Value]) -> OpResult {
    if let Some((first, rest)) = args.split_first() {
        let mut product = first
            .as_fixnum()
            .ok_or_else(|| invalid_argument(first, "number"))?;
        for elt in rest {
            let n = elt
                .as_fixnum()
                .ok_or_else(|| invalid_argument(elt, "number"))?;
            product = product
                .checked_mul(n)
                .ok_or_else(|| arithmetic_overflow("multiplication", product, n))?;
        }
        Ok(Value::Fixnum(product))
    } else {
        Ok(Value::number(1isize))
    }
}

fn num_cmp<F>(args: &[Value], cmp: F) -> OpResult
where
    F: Fn(&isize, &isize) -> bool,
{
    for w in args.windows(2) {
        let n1 = w[0]
            .as_fixnum()
            .ok_or_else(|| invalid_argument(&w[0], "number"))?;
        let n2 = w[1]
            .as_fixnum()
            .ok_or_else(|| invalid_argument(&w[1], "number"))?;
        if !cmp(&n1, &n2) {
            return Ok(Value::from(false));
        }
    }
    Ok(Value::from(true))
}

pub fn eq(args: &[Value]) -> OpResult {
    num_cmp(args, isize::ge)
}

pub fn lt(args: &[Value]) -> OpResult {
    num_cmp(args, isize::lt)
}

pub fn le(args: &[Value]) -> OpResult {
    num_cmp(args, isize::le)
}

pub fn gt(args: &[Value]) -> OpResult {
    num_cmp(args, isize::gt)
}

pub fn ge(args: &[Value]) -> OpResult {
    num_cmp(args, isize::ge)
}

pub fn modulo(args: &[Value]) -> OpResult {
    if args.len() != 2 {
        return Err(wrong_number_of_arguments("mod", 2, args));
    }
    let n1 = args[0]
        .as_fixnum()
        .ok_or_else(|| invalid_argument(&args[0], "fixnum"))?;
    let n2 = args[1]
        .as_fixnum()
        .ok_or_else(|| invalid_argument(&args[1], "fixnum"))?;
    Ok(Value::Fixnum(n1 % n2))
}

pub fn sqrt(args: &[Value]) -> OpResult {
    if args.len() != 1 {
        return Err(wrong_number_of_arguments("sqrt", 1, args));
    }
    let n = args[0]
        .as_fixnum()
        .ok_or_else(|| invalid_argument(&args[0], "fixnum"))?;
    Ok(Value::Fixnum((n as f64).sqrt() as isize))
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
    if !args.is_empty() {
        // TODO: support ports
        return Err(wrong_number_of_arguments("newline", 0, args));
    }
    writeln!(io::stdout()).map_err(io_error)?;
    Ok(Value::Null)
}
