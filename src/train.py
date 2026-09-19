from src.utils import current_millis, format_seconds


def run_one_iteration(model) -> None:
    for u in range(model.user_count):
        model.update_user(u)

    for i in range(model.item_count):
        model.update_item(i)


def train_model(model) -> None:
    previous_loss = float("inf")

    for iteration in range(model.config.max_iter):
        start = current_millis()

        run_one_iteration(model)

        elapsed_seconds = (current_millis() - start) / 1000.0

        if model.config.show_progress:
            print(f"Iteration {iteration + 1}/{model.config.max_iter} completed in {format_seconds(elapsed_seconds)}")

        if model.config.show_loss:
            current_loss = model.loss()
            if current_loss is None:
                print("Current loss: None")
            else:
                symbol = "-" if current_loss <= previous_loss else "+"
                print(f"Current loss: {current_loss:.6f} [{symbol}]")
                previous_loss = current_loss