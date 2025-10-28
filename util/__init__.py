def cfg_rewriter(configs, keyword, values):
    assert len(configs) == len(values)
    for cfg_idx, cfg in enumerate(configs):
        if not hasattr(cfg, keyword):
            raise ValueError(cfg, keyword, values)
        setattr(configs[cfg_idx], keyword, values[cfg_idx])