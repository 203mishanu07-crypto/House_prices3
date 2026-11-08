import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import pandas as pd
import shap

class Drawer():
# Отбирает из результатов все значения с целевым, а остальные по указаному фиксу
    @staticmethod
    def draw_parameter(results, parameter, fixed_params=None):
        column = f"param_{parameter}"

        data = results.copy()
        data[column] = data[column].astype(float)
        data = data.sort_values(column)

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.errorbar(
            x=data[column],
            y=data["mean_val_score"],
            yerr=data["std_val_score"],
            marker="o",
        )

        ax.set_xlabel(parameter)
        ax.set_ylabel("CV RMSE")
        ax.set_title(f"RMSE vs {parameter}")
        ax.grid(alpha=0.3)

        return fig

    @staticmethod
    def draw_heatmap(results, parameter_x, parameter_y, fixed_params=None):
        data = results.copy()

        data[parameter_x] = data[f"param_{parameter_x}"].astype(float)
        data[parameter_y] = data[f"param_{parameter_y}"].astype(float)

        duplicates = data.duplicated(
            subset=[parameter_x, parameter_y]
        ).sum()

        if duplicates > 0:
            print(
                f"Найдено {duplicates} повторов пар параметров. "
                "Для каждой пары будет рассчитан средний CV RMSE."
            )

        heatmap_data = data.pivot_table(
            index=parameter_y,
            columns=parameter_x,
            values="mean_val_score",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        if heatmap_data.empty:
            raise ValueError("Нет данных для построения heatmap.")

        fig, ax = plt.subplots(figsize=(14, 8))

        heatmap = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": "Mean CV RMSE"},
        )

        ax.set_xlabel(parameter_x)
        ax.set_ylabel(parameter_y)
        ax.set_title(f"CV RMSE: {parameter_y} vs {parameter_x}")

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
