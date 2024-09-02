pub fn get_shape<T>(vec: &Vec<T>) -> Vec<usize> {
    if vec.is_empty() {
        return vec![];
    }

    let mut shape = vec![vec.len()];
    add_shape(&vec[0], &mut shape);
    shape
}

fn add_shape<T>(element: &T, shape: &mut Vec<usize>) {
    if let Some(inner_vec) = element_as_vec(element) {
        if !inner_vec.is_empty() {
            shape.push(inner_vec.len());
            add_shape(&inner_vec[0], shape);
        }
    }
}

fn element_as_vec<T>(element: &T) -> Option<&Vec<T>> {
    // Using unsafe code to cast element to Vec<T> if possible
    unsafe {
        let ptr = element as *const T as *const Vec<T>;
        if (*ptr).is_empty() {
            None
        } else {
            Some(&*ptr)
        }
    }
}

