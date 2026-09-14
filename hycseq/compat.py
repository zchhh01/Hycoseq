from .data import build_loaders
from .runtime import build_model, build_optimizer, load_checkpoint as restore_state
from .settings import ExperimentSettings
from .recurrent import ensure_recurrent_inactive
from .tokens import BASES, resolve_token_layout


def ensure_input_args(args):
    representation_map = {
        "nucleotide": "base",
        "kmer": "kmer_onehot",
        "kmer_embedding": "kmer_embed",
    }
    old_representation = str(getattr(args, "input_encoding", "nucleotide")).lower()
    if old_representation not in representation_map:
        raise ValueError("unsupported legacy input_encoding: " + old_representation)
    layout = resolve_token_layout(
        representation_map[old_representation],
        int(args.length),
        int(getattr(args, "kmer_size", 3)),
        int(getattr(args, "kmer_stride", 1)),
        int(getattr(args, "kmer_embedding_dim", 64)),
    )
    args.input_encoding = old_representation
    args.input_dim = layout.input_width
    args.token_length = layout.sequence_steps
    args.kmer_vocab_size = layout.vocabulary_size
    args.kmer_unk_id = layout.unknown_id
    args.kmer_pad_id = layout.padding_id
    args.kmer_alphabet = ",".join(BASES)
    args.kmer_n_policy = str(getattr(args, "kmer_n_policy", "unk"))
    args.kmer_embedding_dim = int(getattr(args, "kmer_embedding_dim", 64))
    args.debug_shapes = bool(getattr(args, "debug_shapes", False))
    args.rc_aug = bool(getattr(args, "rc_aug", False))
    return args


def _settings(args):
    ensure_recurrent_inactive(
        bool(getattr(args, "use_bilstm_context", False)),
        bool(getattr(args, "use_bilstm_lresnet", False)),
    )
    args = ensure_input_args(args)
    if str(getattr(args, "manifold", "lorentz")).lower() != "lorentz":
        raise ValueError("the compatibility entry point now supports only the Lorentz backbone")
    if str(getattr(args, "residual_type", "lresnet")).lower() != "lresnet":
        raise ValueError("the space-like residual was removed; set residual_type=lresnet")
    representation = {"nucleotide": "base", "kmer": "kmer_onehot", "kmer_embedding": "kmer_embed"}[args.input_encoding]
    weighting = "adaptive" if str(getattr(args, "lresnet_weight_mode", "fixed")).lower() == "learnable" else "fixed"
    scale_mode = "adaptive" if str(getattr(args, "lresnet_scale_mode", "fixed")).lower() == "learnable" else "fixed"
    layout = resolve_token_layout(representation, args.length, args.kmer_size, args.kmer_stride, args.kmer_embedding_dim)
    return ExperimentSettings(
        run_id=args.run_name,
        corpus=args.benchmark,
        task=args.dataset_name,
        data_root=args.data_path,
        save_root=args.output_dir,
        accelerator=args.device,
        seed=args.seed,
        class_count=args.num_classes,
        epochs=args.num_epochs,
        batch=args.batch_size,
        max_bases=args.length,
        resume_from=args.load_checkpoint,
        learning_rate=args.lr,
        decay=args.weight_decay,
        geometry_learning_rate=args.manifold_lr,
        geometry_decay=args.manifold_weight_decay,
        optimizer_name=args.optimizer,
        schedule=args.use_lr_scheduler,
        schedule_steps=args.lr_scheduler_milestones,
        schedule_factor=args.lr_scheduler_gamma,
        width=args.num_channels,
        stages=args.num_layers,
        head_width=args.embedding_dim,
        layerwise_curvature=args.multi_k_model,
        learn_curvature=args.learnable_k,
        curvature_scale=args.k,
        merge_weights=weighting,
        skip_weight=args.lresnet_wx,
        transform_weight=args.lresnet_wy,
        transform_weight_init=args.lresnet_wy_init,
        rescale_output=args.lresnet_use_scale,
        rescale_mode=scale_mode,
        rescale_init=args.lresnet_scale_init,
        representation=representation,
        kmer=args.kmer_size,
        token_stride=args.kmer_stride,
        ambiguous_policy="unknown",
        token_width=args.kmer_embedding_dim,
        reverse_complement=args.rc_aug,
        shape_trace=args.debug_shapes,
        layout=layout,
    )


def select_model(args):
    return build_model(_settings(args))


def select_dataset(args):
    return build_loaders(_settings(args))


def select_optimizer(model, args):
    return build_optimizer(model, _settings(args))


def load_checkpoint(model, optimizer, lr_scheduler, args):
    restore_state(args.load_checkpoint, model, optimizer, lr_scheduler)
    return model, optimizer, lr_scheduler
