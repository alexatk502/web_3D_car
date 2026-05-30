// Native smoke test of the public World API (Phase 0: soft-body core).
// Rendering/WebGPU is browser-only.

use web_3d_car::World;

#[test]
fn world_builds_and_steps() {
    let mut world = World::new();

    // Static scene descriptor: terrain + 10 obstacle boxes = 11 objects.
    let desc = world.descriptor();
    assert!(desc.starts_with('['), "descriptor should be a JSON array");
    assert!(desc.contains("\"terrain\""), "should include rocky terrain");
    assert!(desc.contains("\"box\""), "should include obstacle boxes");
    // 11 static (terrain + 10 obstacles) + 4 wheels.
    assert_eq!(world.object_count(), 15);
    assert_eq!(world.wheel_count(), 4);
    assert_eq!(world.buffer_len(), 16 * (world.object_count() + 1));

    // Car descriptor carries wheel + cage info for rendering.
    let car = world.car_descriptor();
    assert!(car.contains("\"wheelRadius\"") && car.contains("\"cageMin\""));

    // Soft body present and wired to the node buffer.
    assert!(world.node_count() > 0, "soft body should have nodes");
    assert_eq!(world.node_buffer_len(), world.node_count() * 3);

    let soft = world.soft_descriptor();
    assert!(soft.contains("\"beams\""), "soft descriptor should list beams");
    assert!(soft.contains("\"nodeCount\""));

    // Step ~2 seconds; nothing should panic and buffers stay valid.
    for _ in 0..120 {
        world.step(1.0 / 60.0, 0, 0.0, 0.0);
    }
    assert!(!world.buffer_ptr().is_null());
    assert!(!world.node_buffer_ptr().is_null());

    // After settling, average node speed should be small and finite.
    let s = world.speed_kmh();
    assert!(s.is_finite() && s < 50.0, "speed should be finite/settled: {}", s);
}
