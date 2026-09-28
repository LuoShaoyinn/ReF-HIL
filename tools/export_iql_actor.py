"""Export a saved IQL actor for ordinary actor.validate; no SAC or fence."""
import argparse
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from learner.action_proximity import RecoveryActorHead
from shared.actor_network import ActorInferencePolicy, ActorNetworkConfig, ObservationEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output_dir / args.checkpoint.name
    if output.exists():
        raise FileExistsError(output)
    with args.checkpoint.open('rb') as handle:
        ck = pickle.load(handle)
    state = ck['policy']
    config = ActorNetworkConfig(**ck['actor']['config'])
    encoder = {k: torch.as_tensor(v).cpu() for k, v in state['recovery_encoder'].items()}
    original = {k: torch.as_tensor(v).cpu() for k, v in state['recovery_actor'].items()}
    mapped = {('continuous.' + k[len('output.'):] if k.startswith('output.') else k): v.numpy()
              for k, v in original.items()}
    snapshot = dict(format='deterministic_actor_v1', config=config.as_dict(),
                    encoder={k: v.numpy() for k, v in encoder.items()}, actor=mapped,
                    output_activation='tanh')
    reference_encoder = ObservationEncoder(config.obs_dims, config.mechanism_obs_dims, config.encoder_dim)
    reference = RecoveryActorHead(reference_encoder.output_dim, config.hidden_dim, config.action_dims)
    reference_encoder.load_state_dict(encoder); reference.load_state_dict(original)
    policy = ActorInferencePolicy(config=config, device='cpu'); policy.load(snapshot)
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        observations = torch.randn(128, config.obs_dims, generator=generator)
        expected = reference(reference_encoder(observations))
        actual = policy.actor(policy.encoder(observations))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    export = dict(format_version=ck['format_version'], class_name='ExportedIQLActor',
                  elapsed_seconds=ck['elapsed_seconds'], learner=ck['learner'], actor=snapshot,
                  source_checkpoint=str(args.checkpoint.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        pickle.dump(export, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Exported {output}; exact IQL action parity verified on 128 inputs.')


if __name__ == '__main__':
    main()
