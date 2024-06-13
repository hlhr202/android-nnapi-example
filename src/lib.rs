use android_activity::AndroidApp;
mod context;
mod primitives;
mod tensor;
mod test;

#[no_mangle]
fn android_main(_app: AndroidApp) {
    android_logger::init_once(android_logger::Config::default().with_max_level(log::max_level()));

    std::thread::sleep(std::time::Duration::from_secs(1));

    std::thread::spawn(|| test::test_nn().unwrap());

    std::thread::sleep(std::time::Duration::from_secs(3));
}
