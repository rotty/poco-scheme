use super::*;
use lexpr::sexp;

fn assert_ref(ast: &Ast, name: &str, value: Value) {
    match ast {
        Ast::Ref(env_ref) => {
            assert_eq!(env_ref.ident.as_ref(), name);
            assert_eq!(env_ref.cell.name.as_ref(), name);
            assert_eq!(*env_ref.cell.value.borrow(), value);
        }
        _ => panic!("unexpected AST, expected ref: {:?}", ast),
    }
}

#[test]
fn check_analyze_datum() {
    let ctx = Context::default();
    for datum in &[sexp!(1), sexp!(#t), sexp!(#f)] {
        let ast = ctx.analyze(datum).expect("analyze failed");
        assert_eq!(ast.as_datum(), Some(datum));
    }
}

#[test]
fn check_analyze_var_ref() {
    let ctx = Context::default();
    let foo = ctx.analyze(&sexp!(foo)).expect("analyze failed");
    assert_ref(&foo, "foo", Value::Undefined);
}

#[test]
fn check_analyze_begin() {
    let ctx = Context::make_eval();
    let seq = ctx
        .analyze(&sexp!((begin 1 foo 3)))
        .expect("analyze failed");
    match seq {
        Ast::Seq(elts, last) => {
            assert_eq!(elts.len(), 2);
            assert_eq!(elts[0].as_datum(), Some(&sexp!(1)));
            assert_eq!(last.as_datum(), Some(&sexp!(3)));
        }
        _ => panic!("unexpected AST: {:?}", seq),
    }
}

#[test]
fn check_analyze_app() {
    let ctx = Context::make_eval();
    let seq = ctx.analyze(&sexp!((foo 23 bar))).expect("analyze failed");
    match seq {
        Ast::Apply(apply) => {
            assert_ref(&apply.op, "foo", Value::Undefined);
            let rands = apply.operands;
            assert_eq!(rands.len(), 2);
            assert_eq!(rands[0].as_datum(), Some(&sexp!(23)));
            assert_ref(&rands[1], "bar", Value::Undefined);
        }
        _ => panic!("unexpected AST: {:?}", seq),
    }
}

#[test]
fn check_analyze_define_var() {
    let ctx = Context::make_eval();
    let ast = ctx
        .analyze(&sexp!((define bar 23)))
        .expect("analyze failed");
    match ast {
        Ast::Set(env_ref, value) => {
            assert_eq!(env_ref.ident.as_ref(), "bar");
            assert_eq!(value.as_datum(), Some(&sexp!(23)));
        }
        _ => panic!("unexpected AST: {:?}", ast),
    }
}

#[test]
fn check_analyze_define_proc() {
    let ctx = Context::make_eval();
    let ast = ctx
        .analyze(&sexp!((define (bar x y) (#"+" x y))))
        .expect("analyze failed");
    match ast {
        Ast::Set(env_ref, value) => {
            assert_eq!(env_ref.ident.as_ref(), "bar");
            let lambda = value.as_lambda().expect("non-lambda value");
            assert_eq!(
                lambda.borrow().name.as_ref().map(Rc::clone),
                Some("bar".into())
            );
        }
        _ => panic!("unexpected AST: {:?}", ast),
    }
    let _ = ctx.env_ref("bar").expect("bar not defined");
}

#[test]
fn check_analyze_lambda() {
    let ctx = Context::make_eval();
    let l = ctx.analyze(&sexp!((lambda (x) x))).expect("analyze failed");
    match l {
        Ast::Lambda(lambda) => {
            use std::ops::Deref;
            let lambda = lambda.borrow();
            assert_eq!(lambda.params, Params::Exact(vec!["x".into()]).into());
            assert_eq!(lambda.name, None);
            assert!(lambda.defs.is_empty());
            assert!(lambda.locals.is_empty());
            let body = lambda.body.as_ref().unwrap();
            match body.deref() {
                Ast::Ref(env_ref) => {
                    assert_eq!(env_ref.ident.as_ref(), "x");
                }
                _ => panic!("unexpected body: {:?}", lambda.body),
            }
        }
        _ => panic!("unexpected value: {:?}", l),
    }
}
