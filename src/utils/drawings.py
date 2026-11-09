import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import pandas as pd
import shap

class Drawer():
# Отбирает из результатов все значения с целевым, а остальные по указаному фиксу
    @staticmethod
    def _filter_fixed_params(results, fixed_params, varying_params):
        if fixed_params is None:
            raise ValueError("Передай лучшие параметры в fixed_params.")

        data = results.copy()

        for parameter, value in fixed_params.items():
            if parameter in varying_params:
                continue

            column = f"param_{parameter}"

            if column not in data.columns:
                raise ValueError(
                    f"Нельзя зафиксировать {parameter}: "
                    f"столбец {column} отсутствует."
                )

            if isinstance(value, (list, tuple, np.ndarray)):
                mask = data[column].map(
                    lambda x: (
                        np.array_equal(x, value)
                        if isinstance(x, (list, tuple, np.ndarray))
                        else False
                    )
                )
            elif value is None:
                mask = data[column].isna()
            elif np.isscalar(value) and pd.isna(value):
                mask = data[column].isna()
            else:
                mask = data[column].map(
                    lambda x: (
                        False
                        if isinstance(x, (list, tuple, np.ndarray))
                        else x == value
                    )
                ).fillna(False)

            data = data.loc[mask].copy()

        if data.empty:
            raise ValueError(
                "После фиксации остальных параметров "
                "не осталось результатов."
            )

        return data

    @staticmethod
    def draw_parameter(results, parameter, fixed_params=None):
        data = Drawer._filter_fixed_params(
            results,
            fixed_params,
            varying_params={parameter},
        )

        column = f"param_{parameter}"
        data[column] = pd.to_numeric(data[column], errors="raise")
        data = data.sort_values(column)

        if data.duplicated(subset=[column]).any():
            raise ValueError(
                "Одному значению параметра соответствуют несколько "
                "результатов. Проверь повторные конфигурации "
                "и полноту fixed_params."
            )

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.errorbar(
            x=data[column],
            y=data["mean_val_score"],
            yerr=data["std_val_score"],
            marker="o",
        )

        ax.set_xlabel(parameter)
        ax.set_ylabel("CV RMSE")
        ax.set_title(f"RMSE vs {parameter}; остальные параметры фиксированы")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        return fig

    @staticmethod
    def draw_heatmap(results, parameter_x, parameter_y, fixed_params=None):
        if parameter_x == parameter_y:
            raise ValueError("Для осей выбери разные параметры.")

        data = Drawer._filter_fixed_params(
            results,
            fixed_params,
            varying_params={parameter_x, parameter_y},
        )

        column_x = f"param_{parameter_x}"
        column_y = f"param_{parameter_y}"

        data[column_x] = pd.to_numeric(data[column_x], errors="raise")
        data[column_y] = pd.to_numeric(data[column_y], errors="raise")

        if data.duplicated(subset=[column_x, column_y]).any():
            raise ValueError(
                "Одной паре параметров соответствуют несколько "
                "результатов. Проверь повторные конфигурации "
                "и полноту fixed_params."
            )

        heatmap_data = (
            data.pivot(
                index=column_y,
                columns=column_x,
                values="mean_val_score",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        heatmap = sns.heatmap(
            heatmap_data,
            mask=heatmap_data.isna(),
            annot=True,
            fmt=".4f",
            cmap="viridis_r",
            ax=ax,
            cbar_kws={"label": "Mean CV RMSE"},
        )

        ax.set_xlabel(parameter_x)
        ax.set_ylabel(parameter_y)
        ax.set_title("CV RMSE; остальные параметры фиксированы")
        fig.tight_layout()

        return heatmap


    @staticmethod
    def draw_graphics(results_df,oof_rmse,pipeline,best_results,model):
        print(f'Best OOF {oof_rmse}, best mean val RMSE {best_results['best_mean_val_score']}, best_val_std {best_results['best_val_std']}')
        title = 'Best searched params'
        print(title.center(20, '-'))
        best_results_df = pd.DataFrame([best_results['best_params']])

        if 'model__num_cols' in best_results_df.columns:
            best_results_df = best_results_df.drop('model__num_cols',axis = 1)
        if 'model__cat_cols' in best_results_df.columns:
            best_results_df = best_results_df.drop('model__cat_cols',axis = 1)
        
        print(best_results_df.T.to_string(header=False))
        
        fig1 = Drawer.draw_parameter(results_df,pipeline[model]['plot_param'],best_results['best_params'])
        fig2 = Drawer.draw_heatmap(results=results_df,
                                parameter_x=pipeline[model]['heatmap_params'][0],
                                parameter_y=pipeline[model]['heatmap_params'][1],
                                fixed_params=best_results['best_params'])

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        for ax, source in zip(axes, [fig1, fig2]):
            source_fig = source.figure
            source_fig.canvas.draw()

            ax.imshow(source_fig.canvas.buffer_rgba())
            ax.axis("off")

            plt.close(source_fig)

        fig.tight_layout()
        
        return fig

    @staticmethod
    def draw_lerning_curves(histories_by_candidate):
            nplots = len(histories_by_candidate) + 1
            ncols = 2
            nrows = (nplots + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(14, 4 * nrows),
                squeeze=False,
                constrained_layout=True,
            )
            axes = axes.ravel()

            # Метрики каждого фолда.
            for fold_idx, history in enumerate(histories_by_candidate):
                ax = axes[fold_idx]

                ax.plot(history["train_rmse"], label="Train RMSE")
                ax.plot(history["val_rmse"], label="Validation RMSE")

                ax.set_title(f"Fold {history.get('fold', fold_idx + 1)}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("RMSE")
                ax.legend()
                ax.grid(alpha=0.3)

            ax_lr = axes[len(histories_by_candidate)]

            for fold_idx, history in enumerate(histories_by_candidate, start=1):
                ax_lr.plot(
                    history["learning_rate"],
                    label=f"Fold {history.get('fold', fold_idx)}",
                )

            ax_lr.set_title("Learning rate by fold")
            ax_lr.set_xlabel("Epoch")
            ax_lr.set_ylabel("Learning rate")
            ax_lr.set_yscale("log")
            ax_lr.legend()
            ax_lr.grid(alpha=0.3)

            fig.suptitle(
                "Кривые обучения лучшей комбинации параметров",
                fontsize=16,
                fontweight="bold",
            )

            return fig

    @staticmethod
    def draw_shap_plots(oof_explanation,feature_names=None):

        if feature_names is not None:
            oof_explanation.feature_names = feature_names

        plots = {
            "bar": lambda: shap.plots.bar(
                oof_explanation,
                max_display=30,
                show=False
            ),
            "heatmap": lambda: shap.plots.heatmap(
                oof_explanation[:100],
                max_display=30,
                show=False
            )
        }

        figures = {}

        for name, draw in plots.items():
            # plt.close("all")
            draw()
            fig = plt.gcf()

            fig.set_size_inches(16, 8)
            fig.subplots_adjust(left=0.35, right=0.95)

            figures[name] = fig
            display(fig)

            plt.close(fig)
            
        return figures
