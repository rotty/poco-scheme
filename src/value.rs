use std::{
    fmt::{self, Display, Write},
    rc::Rc,
};

use gc::{Finalize, Gc, GcCell};

use crate::{ast::Lambda, evaluator::Env, OpResult};

#[derive(Clone)]
pub enum Value {
    Fixnum(i64),
    String(Box<str>),
    Bool(bool),
    Null,
    Cons(Gc<[Value; 2]>),
    Symbol(Box<str>), // TODO: interning
    PrimOp(&'static str, fn(&[Value]) -> OpResult),
    Closure(Box<Closure>),
}

#[derive(Clone)]
pub struct Closure {
    pub lambda: Rc<Lambda>,
    pub env: Gc<GcCell<Env>>,
}

impl Value {
    pub fn prim_op(name: &'static str, f: fn(&[Value]) -> OpResult) -> Self {
        Value::PrimOp(name, f)
    }

    pub fn list<I>(_elts: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Value>,
    {
        unimplemented!()
    }

    pub fn number<T>(n: T) -> Self
    where
        T: Into<i64>,
    {
        Value::Fixnum(n.into())
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Fixnum(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_true(&self) -> bool {
        if let Value::Bool(v) = self {
            *v
        } else {
            true
        }
    }
    pub fn to_datum(&self) -> Option<lexpr::Value> {
        use Value::*;
        match self {
            Null => Some(lexpr::Value::Null),
            Bool(b) => Some((*b).into()),
            Fixnum(n) => Some((*n).into()),
            String(s) => Some(s.as_ref().into()),
            Symbol(s) => Some(lexpr::Value::Symbol(s.clone())),
            Cons(cell) => {
                let cell = &*cell;
                match (cell[0].to_datum(), cell[1].to_datum()) {
                    (Some(car), Some(cdr)) => Some((car, cdr).into()),
                    _ => None,
                }
            }
            PrimOp(_, _) | Closure(_) => None,
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<&lexpr::Value> for Value {
    fn from(v: &lexpr::Value) -> Self {
        use lexpr::Value::*;
        match v {
            Bool(b) => Value::Bool(*b),
            Number(n) => if let Some(n) = n.as_i64() {
                Value::Fixnum(n)
            } else {
                unimplemented!()
            }
            String(s) => Value::String(s.clone()),
            Symbol(s) => Value::Symbol(s.clone()),
            Cons(cell) => {
                let (car, cdr) = cell.as_pair();
                Value::Cons(Gc::new([car.into(), cdr.into()]))
            }
            Null => Value::Null,
            _ => unimplemented!(),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Should probably use a more "Rusty" representation
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Fixnum(n) => write!(f, "{}", n),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Bool(b) => f.write_str(if *b { "#t" } else { "#f" }),
            Value::PrimOp(name, _) => write!(f, "#<prim-op {}>", name),
            Value::Closure { .. } => write!(f, "#<closure>"),
            Value::Null => write!(f, "()"),
            Value::Cons(cell) => write_cons(f, cell),
            Value::String(s) => lexpr::Value::string(s.as_ref()).fmt(f),
        }
    }
}

fn write_cons(f: &mut fmt::Formatter, cell: &[Value; 2]) -> fmt::Result {
    f.write_char('(')?;
    cell[0].fmt(f)?;
    let mut next = &cell[1];
    loop {
        match next {
            Value::Null => break,
            Value::Cons(cell) => {
                f.write_char(' ')?;
                cell[0].fmt(f)?;
                next = &cell[1];
            }
            value => {
                f.write_str(" . ")?;
                value.fmt(f)?;
                break;
            }
        }
    }
    f.write_char(')')?;
    Ok(())
}

impl gc::Finalize for Value {
    fn finalize(&self) {}
}

macro_rules! impl_value_trace_body {
    ($this:ident, $method:ident) => {
        match $this {
            Value::Cons(cell) => {
                cell[0].$method();
                cell[1].$method();
            }
            Value::Closure(boxed) => {
                let Closure  { env, .. } = boxed.as_ref();
                env.$method();
            }
            _ => {}
        }
    };
}

unsafe impl gc::Trace for Value {
    unsafe fn trace(&self) {
        impl_value_trace_body!(self, trace);
    }
    unsafe fn root(&self) {
        impl_value_trace_body!(self, root);
    }
    unsafe fn unroot(&self) {
        impl_value_trace_body!(self, unroot);
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_value_trace_body!(self, finalize_glue);
    }
}
