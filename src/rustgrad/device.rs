use core::panic;
use std::{collections::HashMap, env, fs, ops::Index, path::Path, rc::Rc};

use super::shape::sym::{sym_infer, BTypes, NodeTypes};

pub struct _Device{
    devices: Vec<String>
}

impl _Device{
    fn new() -> Self{
        let mut devices = Vec::new();
        if let Ok(entries) = fs::read_dir("runtime") {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Some(file_name) = entry.file_name().to_str() {
                        if file_name.starts_with("ops_") {
                            if let Some(stem) = entry.file_name().to_str().map(|s| s["ops_".len()..].to_uppercase()) {
                                devices.push(stem);
                            }
                        }
                    }
                }
            }
        }
        _Device { devices }
    }

    fn _canonicalize(&self, device: &str) -> String
{
    let device = device;
    let mut parts = device.splitn(2, ':');
    let first_part = parts.next().unwrap().to_uppercase();
    let second_part = parts.next().unwrap_or("");
    
    let canonicalized = if device.contains(':') {
        format!("{}:{}", first_part, second_part)
    } else {
        first_part
    };

    canonicalized.replace(":0", "")
}

    fn canonicalize(&self, device: Option<&str>) -> String
    {
        if let Some(d) = device{
            self._canonicalize(d)
        }else{
            _Device::DEFAULT()
        }
    }

    fn __get_canonicalize_item(self, ix: &str) -> Option<Compiled>
    {
        let x: String = ix.split(":").next().unwrap().to_uppercase();
        let module_name = format!("tinygrad.runtime.ops_{}", x.to_lowercase());
        
        let mut compiled: Vec<Compiled> = Vec::new();
        
        if let Ok(module) = importlib::import_module(&module_name) {
            let members = inspect::getmembers(&module);
            for (cname, cls) in members {
                let cname_str: String = cname.try_into().unwrap();
                if cname_str.to_lowercase() == format!("{}device", x.to_lowercase()) && self._devices.contains(&x) {
                    let cls: Compiled = cls.try_into().unwrap();
                    compiled.push(cls);
                }
            }
        }
        
        if compiled.is_empty(){
            None
        }else{
            Some(compiled.pop().unwrap())
        }
    }

    pub fn default(&self) -> String{
        let device_from_env = self.devices.iter().fold(None, |val, ele| {
            if let Ok(value) = env::var(ele) {
                if value == "1" {
                    Some(ele.to_string())
                } else {
                    val
                }
            } else {
                val
            }
        });

        if let Some(device) = device_from_env {
            return device;
        }

        let devices = ["METAL", "HSA", "CUDA", "GPU", "LLVM", "CLANG"];

        for device in devices.iter() {
            if self.index(*device).is_some(){
                return device.to_string();
            }
        }

        panic!()
    }

    pub fn index(&self, index: &str) -> Option<Compiled>{
       self.__get_canonicalize_item(&self.canonicalize(Some(index)))
    }
}

pub const Device: _Device = _Device::new();


struct JITRunner{
    op_estimate: BTypes,
    mem_estimate: BTypes
}

impl JITRunner{
    fn exec(self, rawbufs: Vec<Buffer>, var_vals: Option<HashMap<Rc<NodeTypes>, isize>>) -> Option<f64>{
        let var_vals = var_vals.unwrap_or_else(HashMap::new);
        let et = self.call((rawbufs, var_vals.clone()));
        if CACHECOLLECTING {
            CacheCollector::add(self, &rawbufs, &var_vals);
        }
        et
    }
}
impl JIT_Call for JITRunner{
    fn call(&self, args: (Vec<Buffer>, HashMap<Rc<NodeTypes>, i32>)) -> Option<f64> {
        panic!()
    }
}
trait JIT_Call{
    fn call(&self, args: (Vec<Buffer>, HashMap<Rc<NodeTypes>, i32>)) -> Option<f64> {
        let (_rawbufs, _var_vals) = args;
        None // Replace with actual implementation
    }
}

fn update_stats(name: &str, op_estimate: BTypes, mem_estimate: BTypes, var_vals: Option<HashMap<Rc<NodeTypes>, isize>>, et: Option<f64>, buf_count: usize, jit: bool, num_kernal: usize, lra: Option<HashMap<String, String>>, device: &str, first_run: bool){
    let v_v = {
        if let Some(v) = var_vals{
            v
        }else{
            HashMap::new()
        }
    };
    let op_estimate = sym_infer(, var_vals)
}
#[derive(Debug)]
struct Compiled{

}

struct Buffer{

}