def test_train_loop_runs_short():
    import train_rl_agent as train_mod
    from train_rl_agent import train

    # Temporarily patch TOTAL_STEPS to a small number
    original_total = train_mod.TOTAL_STEPS
    train_mod.TOTAL_STEPS = 10

    train()

    # Restore the original value
    train_mod.TOTAL_STEPS = original_total
