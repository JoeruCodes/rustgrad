use lazy_static::lazy_static;

use crate::{
    create_context_var,
    helpers::{getenv, ContextVar},
};

lazy_static! {
    pub static ref DEBUG: ContextVar = create_context_var!("DEBUG", 0);
    pub static ref IMAGE: ContextVar = create_context_var!("IMAGE", 0);
    pub static ref BEAM: ContextVar = create_context_var!("BEAM", 0);
    pub static ref NOOPT: ContextVar = create_context_var!("NOOPT", 0);
    pub static ref JIT: ContextVar = create_context_var!("JIT", 1);
    pub static ref WINO: ContextVar = create_context_var!("WINO", 0);
    pub static ref THREEFRY: ContextVar = create_context_var!("THEEFRY", 0);
    pub static ref CACHECOLLECTING: ContextVar = create_context_var!("CACHECOLLECTING", 1);
    pub static ref GRAPH: ContextVar = create_context_var!("GRAPH", 0);
    pub static ref GRAPHPATH: String = getenv("GRAPHPATH".to_string(), Some("/tmp/net".to_owned()));
    pub static ref SAVE_SCHEDULE: ContextVar = create_context_var!("SAVE_SCHEDULE", 0);
    pub static ref RING: ContextVar = create_context_var!("RING", 1);
    pub static ref MULTIOUTPUT: ContextVar = create_context_var!("MULTIOUTPUT", 1);
    pub static ref PROFILE: ContextVar = create_context_var!("PROFILE", 0);
}
