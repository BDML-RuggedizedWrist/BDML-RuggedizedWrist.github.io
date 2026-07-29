from rizon_osc.hud import HudSnapshot, format_hud


def test_hud_shows_force_phase_travel_reduction_and_collision():
    """A visible HUD must expose the current comparative safety state."""
    text = format_hud(
        HudSnapshot(
            phase="PITCH_ONLY",
            force_7_n=14.8,
            force_9_n=15.1,
            arm_7_rad=1.0,
            arm_9_rad=0.4,
            wrist_9_rad=0.6,
            reduction_percent=60.0,
            collision_7="NEAR COLLISION",
            collision_9="CONTACT OK",
        )
    )

    assert "PITCH_ONLY" in text
    assert "14.8 / 15.1 N" in text
    assert "1.000 / 0.400 rad" in text
    assert "60.0%" in text
    assert "NEAR COLLISION / CONTACT OK" in text
