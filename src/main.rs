use std::{
    env, fs,
    io::{self, BufRead},
    path::Path,
};

use poco_scheme::{Context, EvalError, Value};

fn load(ctx: &mut Context, path: impl AsRef<Path>) -> Result<(), EvalError> {
    let file = fs::File::open(path)?;
    let parser = lexpr::Parser::from_reader(file);
    for res in ctx.eval_iter(parser) {
        let _ = res?;
    }
    Ok(())
}

fn main() -> Result<(), EvalError> {
    env_logger::init();

    let args: Vec<_> = env::args_os().skip(1).collect();
    let mut ctx = Context::make_eval();
    if args.is_empty() {
        let input = io::BufReader::new(io::stdin());
        for res in ctx.eval_iter::<_, EvalError>(input.lines().map(|line| Ok(line?.parse()?))) {
            match res {
                Ok(value) => {
                    if value != Value::Unspecified {
                        println!("{}", value);
                    }
                }
                Err(e) => println!("; error: {}", e),
            }
        }
    } else {
        for filename in &args {
            load(&mut ctx, filename)?;
        }
    }
    Ok(())
}
