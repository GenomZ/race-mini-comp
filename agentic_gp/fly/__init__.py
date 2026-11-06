"""Fruit-fly brains that drive the car (Ch7 "Fly Brains").

    from agentic_gp.fly import FlyBrain
    brain = FlyBrain.from_default(scramble_seed=None)    # real FlyWire wiring
    controls, snapshot = brain.step(obs)                  # once per car tick

Modules: ``lif`` (spiking simulator), ``connectome`` (circuit data and controls),
``extract`` (instructor tool that builds the circuit file), ``brain`` (encoders, readout,
calibration), ``viz`` (data for the Fly Lab).  Data license: see ATTRIBUTION.md.
"""
from .brain import EncoderParams, FlyBrain, ReadoutParams, calibrate
from .connectome import Circuit, CircuitNotFoundError, load_circuit, lesioned, mirrored, scrambled
from .lif import LIFNetwork, LIFParams, Spikes

__all__ = [
    "FlyBrain", "EncoderParams", "ReadoutParams", "calibrate",
    "Circuit", "CircuitNotFoundError", "load_circuit", "scrambled", "lesioned", "mirrored",
    "LIFNetwork", "LIFParams", "Spikes",
]
