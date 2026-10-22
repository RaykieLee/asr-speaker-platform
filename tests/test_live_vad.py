from backend.app.speakers.vad import EnergyVAD


def test_energy_gate_accepts_voiced_window_and_rejects_silence():
    gate = EnergyVAD()
    assert not gate.is_speech([0.0] * 16_000, 16_000)
    assert gate.is_speech([0.1] * 16_000, 16_000)
    assert not gate.is_speech([0.1] * 1_600 + [0.0] * 14_400, 16_000)
