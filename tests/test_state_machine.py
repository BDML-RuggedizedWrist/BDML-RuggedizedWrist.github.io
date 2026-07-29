from rizon_osc.state_machine import ContactSupervisor, SafetyMode


def test_contact_loss_only_reacquires_after_100ms():
    supervisor = ContactSupervisor(contact_loss_limit=0.1, reacquire_stable_time=0.05)

    for _ in range(9):
        state = supervisor.update(
            dt=0.01, contact=False, surface_valid=True, measured_force=0.0, contact_phase=True
        )
    assert state.mode is SafetyMode.TRACKING

    for _ in range(2):
        state = supervisor.update(
            dt=0.01, contact=False, surface_valid=True, measured_force=0.0, contact_phase=True
        )
    assert state.mode is SafetyMode.REACQUIRE
    assert state.freeze_path


def test_stable_contact_resumes_after_reacquisition():
    supervisor = ContactSupervisor(contact_loss_limit=0.02, reacquire_stable_time=0.03)
    for _ in range(3):
        supervisor.update(
            dt=0.01, contact=False, surface_valid=True, measured_force=0.0, contact_phase=True
        )
    for _ in range(3):
        state = supervisor.update(
            dt=0.01, contact=True, surface_valid=True, measured_force=12.0, contact_phase=True
        )

    assert state.mode is SafetyMode.TRACKING
    assert not state.freeze_path


def test_invalid_surface_and_hard_force_freeze_path():
    supervisor = ContactSupervisor(hard_force_limit=35.0)

    invalid = supervisor.update(
        dt=0.01, contact=True, surface_valid=False, measured_force=15.0, contact_phase=True
    )
    assert invalid.mode is SafetyMode.INVALID_SURFACE
    assert invalid.freeze_path

    supervisor.reset()
    overload = supervisor.update(
        dt=0.01, contact=True, surface_valid=True, measured_force=36.0, contact_phase=True
    )
    assert overload.mode is SafetyMode.FORCE_HOLD
    assert overload.freeze_path
    assert overload.zero_force_command


def test_non_contact_phase_does_not_accumulate_contact_loss():
    supervisor = ContactSupervisor(contact_loss_limit=0.01)

    state = supervisor.update(
        dt=1.0, contact=False, surface_valid=True, measured_force=0.0, contact_phase=False
    )

    assert state.mode is SafetyMode.TRACKING
    assert state.contact_loss_duration == 0.0
    assert state.reset_force_controller

