pub fn add(left: usize, right: usize) -> usize {
    left + right
}
pub mod helpers;
pub mod prelude;
pub mod shape;
mod tests {
    use std::{collections::HashMap, hash::Hash, os::raw::c_void};

    use deep_flatten::DeepFlattenExt;
    use diskcache_proc_macro::diskcache;
    use get_shape::get_shape;
    use helpers::{diskcache_get, fetch, from_mv, mv_address, parse_fetch_paths, to_mv};
    use memoize::memoize;
    use serde::{Deserialize, Serialize};

    use crate::{
        argfix, create_new_context,
        helpers::{analyze_samples, extract_callers, round_up},
        make_pair,
    };
    use init_c_struct_proc_macro::init_c_struct_t;
    #[test]
    fn test_make_pair_macro() {
        let c_1 = 69;
        assert_eq!(make_pair!(c_1), (69, 69));
        assert_eq!(make_pair!(c_1, cnt = 6), vec!(69, 69, 69, 69, 69, 69));

        let c_2 = (3, 4, 5, 6, 7);
        assert_eq!(make_pair!((c_2)), (3, 4, 5, 6, 7));
        assert_eq!(make_pair!((c_2), cnt = 6), (3, 4, 5, 6, 7));
    }

    #[test]
    fn test_arg_fix() {
        let c_1 = 69;
        assert_eq!(argfix!(c_1, c_1, c_1), (69, 69, 69));

        // argfix!(((69, 69), (69, 69)));
    }

    #[test]
    fn test_deep_flatten() {
        let z = vec![
            vec![],
            vec![vec![1], vec![2, 3]],
            vec![vec![4, 5, 6], vec![7]],
        ]
        .into_iter();
        assert_eq!(
            z.deep_flatten::<_, i32>().collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn test_round_up() {
        let t: isize = round_up::<f64>(10.8, 2.0);
        assert_eq!(t, 10)
    }
    #[test]
    fn test_get_shape_macro() {
        let x = vec![
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            vec![vec![7, 8, 9], vec![10, 11, 12]],
        ];
        assert_eq!(get_shape(&x), vec![2, 2, 3]);
    }
    // #[lru_cache]
    // fn expensive_computation(input: u32) -> u32 {
    //     // Expensive computation here
    //     input * 2
    // }
    #[test]
    fn test_context_create_macro() {
        let x1 = 2;
        let x2 = 1;
    }

    #[test]
    fn test_extract_callers() {
        use pprof::protos::profile::{Function, Line, Location, Sample, ValueType};
        use pprof::protos::Profile;
        // Create mock profile data
        let sample_type = vec![
            ValueType {
                ty: 1,
                unit: 2,
                ..Default::default()
            }, // num_calls
            ValueType {
                ty: 3,
                unit: 4,
                ..Default::default()
            }, // tottime
            ValueType {
                ty: 5,
                unit: 6,
                ..Default::default()
            }, // cumtime
        ];

        let sample = vec![Sample {
            location_id: vec![2, 1], // Caller function 2, Callee function 1
            value: vec![10, 1000, 2000],
            label: vec![].into(),
            ..Default::default()
        }];

        let function = vec![
            Function {
                id: 1,
                name: 7,
                system_name: 8,
                filename: 9,
                start_line: 10,
                ..Default::default()
            },
            Function {
                id: 2,
                name: 10,
                system_name: 11,
                filename: 12,
                start_line: 13,
                ..Default::default()
            },
        ];

        let location = vec![
            Location {
                id: 1,
                mapping_id: 0,
                address: 0,
                line: vec![Line {
                    function_id: 1,
                    line: 0,
                    ..Default::default()
                }]
                .into(),
                is_folded: false,
                ..Default::default()
            },
            Location {
                id: 2,
                mapping_id: 0,
                address: 0,
                line: vec![Line {
                    function_id: 2,
                    line: 0,
                    ..Default::default()
                }]
                .into(),
                is_folded: false,
                ..Default::default()
            },
        ];

        let string_table = vec![
            "".to_string(),
            "num_calls".to_string(),
            "count".to_string(),
            "cpu".to_string(),
            "nanoseconds".to_string(),
            "cumulative_time".to_string(),
            "nanoseconds".to_string(),
            "function_name".to_string(),
            "system_name".to_string(),
            "filename".to_string(),
            "caller_function_name".to_string(),
            "caller_system_name".to_string(),
            "caller_filename".to_string(),
            "callee_function_name".to_string(),
            "callee_system_name".to_string(),
        ];

        let profile = Profile {
            sample_type: sample_type.into(),
            sample: sample.into(),
            mapping: vec![].into(),
            location: location.into(),
            function: function.into(),
            string_table: string_table.into(),
            drop_frames: 0,
            keep_frames: 0,
            time_nanos: 0,
            duration_nanos: 0,
            period_type: None.into(),
            period: 0,
            comment: vec![],
            default_sample_type: 0,
            ..Default::default()
        };

        let mut expected_callers = HashMap::new();
        let mut inner_map = HashMap::new();
        inner_map.insert("function_name".to_string(), 1); // Function 1 called by function 2
        expected_callers.insert("caller_function_name".to_string(), inner_map);

        let callers = extract_callers(&profile);
        assert_eq!(callers, expected_callers);
    }

    use super::*;
    use rusqlite::{Connection, Result};

    // Mock implementation of pickle::loads for testing
    fn mock_pickle_loads(_val: &[u8]) -> Option<String> {
        // Simulate deserialization, for example, converting bytes to String
        Some(String::from_utf8_lossy(_val).to_string())
    }
    #[derive(Deserialize, PartialEq, Debug)]
    struct Test {
        hi: String,
        hello: String,
    }
    #[test]
    fn test_diskcache_get() {
        // Test case 2: Cache level is 0, should return None
        {
            const CACHELEVEL: i32 = 0;
            const VERSION: &str = "v1";

            // Mock key (not used since cache level is 0)

            // Test the macro
            let result: Option<Test> = diskcache_get("table", Box::new("hello"));
            assert_eq!(result, None);
        }
    }

    #[diskcache]
    fn expensive_function(arg1: u32, arg2: String) -> Result<MyType, Error> {
        // Perform some expensive computations
        Result::<MyType, ()>::Ok(MyType {
            a: arg1,
            b: arg2.clone(),
        })
    }

    #[test]
    fn test_expensive() {
        let _ = expensive_function(1, "hello".to_string()).unwrap();
    }
    #[derive(Serialize, Deserialize, Clone)]
    struct MyType {
        a: u32,
        b: String,
    }

    #[tokio::test]
    async fn test_fetch() {
        let ret = fetch(
            "https://google.com/",
            parse_fetch_paths("https://google.com/", None, None),
            None,
        )
        .await
        .unwrap();
    }

    #[test]
    fn test_from_mv() {
        // Example input memory view
        let mv: [u8; 4] = [0x01, 0x02, 0x03, 0x04];

        // Call the function with a specific type
        let ptr = from_mv::<u32>(&mv);

        // Dereference the pointer to get the value
        let value = unsafe { *ptr };

        // Assert the value to match the expected output
        assert_eq!(value, 0x04030201);
    }

    #[test]
    fn test_to_mv() {
        // Simulate a pointer to some memory (for demonstration purposes)
        let data: [u8; 6] = [1, 2, 3, 4, 5, 6];
        let ptr = data.as_ptr();
        let sz = data.len();

        // Call the function under test
        let result = to_mv(ptr as *const std::ffi::c_void, sz);

        // Assert that the result is as expected
        assert_eq!(result.unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_from_to_mv() {
        // Example data to convert
        let data = [1, 2, 3, 4, 5];

        // Convert the data to a memory view pointer
        let ptr = from_mv::<u8>(&data);

        // Convert the pointer back to a memory view
        let mv_opt = to_mv(ptr as *const c_void, data.len());

        // Assert that the conversion was successful and the data matches
        assert!(mv_opt.is_some(), "Conversion to memory view failed");
        let mv = mv_opt.unwrap();
        assert_eq!(
            mv, &data,
            "Converted memory view does not match original data"
        );
    }
    #[test]
    fn test_mv_address() {
        let data = [1, 2, 3, 4, 5];
        let address = mv_address(&data);

        // Verify that address is not null
        assert!(!address.is_null());

        // Check if address retrieved matches pointer from from_mv
        let ptr_from_mv: *const u8 = from_mv(&data);
        assert_eq!(address, ptr_from_mv as *const c_void);

        // Check if to_mv can retrieve the original data
        let sz = data.len();
        let retrieved_data = to_mv(address, sz).unwrap();
        assert_eq!(&data[..], retrieved_data);
    }


#[init_c_struct_t( field1 = i32, field2 = f64, field3 = u8 )]
struct MyStruct{
    a: usize
}
}
