import numpy as np
import pandas as pd
import shap


def count_off_shap(
    pipelines, mode, X, run_id, cv_folds, background_size=50, n_permutations=5, seed=42
):
    """Рассчитывает SHAP-важности на валидационных частях фолдов.

    Для каждой обученной модели объясняет её прогнозы на соответствующем
    валидационном фолде. При необходимости фон выбирается из тренировочной
    части. Результаты объединяются по позициям строк и именам признаков.

    Args:
        pipelines: Обученные пайплайны в порядке cv_folds.
        mode: Режим выбора SHAP explainer.
        X: Исходный DataFrame, на котором были построены разбиения.
        run_id: Идентификатор эксперимента
        cv_folds: Список пар позиционных индексов (train_idx, val_idx).
        background_size: Максимальное число фоновых объектов на фолд.
        n_permutations: Зарезервированный параметр; в текущей реализации
            не управляет количеством перестановок.
        seed: Seed для выбора фоновых объектов.

    Returns:
        tuple: Таблица средней абсолютной SHAP-важности по признакам
            и объединённый объект shap.Explanation.
    """
    cv_folds = list(cv_folds)
    supported_modes = {"catboost", "lightgbm", "tree", "linear", "dnn"}

    all_features = sorted(
        set().union(*[pipeline[:-1].get_feature_names_out() for pipeline in pipelines])
    )

    oof_shap = pd.DataFrame(0.0, index=np.arange(len(X)), columns=all_features)

    if mode not in supported_modes:
        raise ValueError(f"Неизвестный режим SHAP: {mode!r}")

    for fold_idx, (pipeline, split) in enumerate(
        zip(pipelines, cv_folds),
        start=1,
    ):
        train_idx, val_idx = split

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        # На чем будем объяснять
        X_background = X_train.sample(
            n=min(background_size, len(train_idx)),
            random_state=seed,
        )

        # Делим пайплайн
        transformer = pipeline[:-1]
        model = pipeline[-1]

        # Получаем именна колонок
        feature_names = transformer.get_feature_names_out()
        # Преобразуем данные
        X_background_transformed = transformer.transform(X_background)
        X_val_transformed = transformer.transform(X_val)

        # Достаем имена колонок из разных видов трансформеров
        if isinstance(X_background_transformed, pd.DataFrame):
            transformed_names = X_background_transformed.columns.tolist()
        else:
            transformed_names = transformer.get_feature_names_out().tolist()

        X_val_transformed = pd.DataFrame(
            X_val_transformed, columns=transformed_names, index=X_val.index
        )

        masker = shap.maskers.Independent(X_background_transformed)

        if mode == "linear":
            explainer = shap.LinearExplainer(model, masker)
        elif mode == "dnn":
            explainer = shap.PermutationExplainer(model.predict, masker)
        elif mode in ["catboost", "lightgbm", "xgboost"]:
            explainer = shap.TreeExplainer(
                model, feature_perturbation="tree_path_dependent"
            )
        elif mode in ["tree", "random_forest"]:
            explainer = shap.TreeExplainer(model)
        elif mode == "knn":
            explainer = shap.PermutationExplainer(model.predict, masker)

        sv = explainer(X_val_transformed)

        oof_shap.loc[val_idx, feature_names] = sv.values

    importance = pd.DataFrame(
        {"feature": all_features, "importance": np.abs(oof_shap).mean(axis=0)}
    ).sort_values("importance", ascending=False)

    oof_explanation = shap.Explanation(
        values=oof_shap.to_numpy(), feature_names=oof_shap.columns.tolist()
    )

    return importance, oof_explanation  # Для графиков
