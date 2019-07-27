use std::fmt::{self, Write};

pub struct ShowSlice<'a, T>(pub &'a [T]);

impl<'a, T> fmt::Display for ShowSlice<'a, T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, value) in self.0.iter().enumerate() {
            value.fmt(f)?;
            if i + 1 < self.0.len() {
                f.write_char(' ')?;
            }
        }
        Ok(())
    }
}
