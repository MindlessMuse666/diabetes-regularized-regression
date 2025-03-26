import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings


PLOTS_DIR = 'plots//'


class RegressionModelTrainer:
    '''
    Класс для обучения и оценки регрессионных моделей с регуляризацией.
    '''
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()


    def load_data(self):
        '''
        Загружает данные из CSV-файла и выполняет предварительную обработку.
        '''
        try:
            self.data = pd.read_csv(self.data_path)
            print('Данные успешно загружены.')
        except FileNotFoundError:
            print(f'Ошибка: Файл не найден по пути {self.data_path}')
            return
        except Exception as e:
            print(f'Ошибка при загрузке данных: {e}')
            return

        # Обработка пропущенных значений (заполнение медианой)
        for col in self.data.columns:
            if self.data[col].isnull().any():
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                print(f'Пропущенные значения в столбце {col} заполнены медианой ({median_val}).')

        # Вывод информации о данных
        print('\nИнформация о данных:')
        self.data.info()
        print('\nПервые 5 строк данных:')
        print(self.data.head())
        print('\nОписательная статистика:')
        print(self.data.describe())


    def visualize_data(self):
        '''
        Визуализирует данные. Строит гистограммы и матрицу корреляции.
        '''
        if self.data is None:
            print('Ошибка: Данные не загружены. Сначала загрузите данные.')
            return

        # Гистограммы для каждого столбца
        num_cols = len(self.data.columns)
        num_rows = (num_cols + 2) // 3
        plt.figure(figsize=(9, 3 * num_rows), facecolor='lightgrey')
        plt.suptitle('Гистограммы распределения признаков', fontsize=16, fontweight='bold')

        for i, column in enumerate(self.data.columns, 1):
            plt.subplot(num_rows, 3, i)
            sns.histplot(self.data[column], kde=True, palette='viridis')
            plt.title(f'Распределение {column}', fontsize=10)
            plt.xlabel(column, fontsize=8)
            plt.ylabel('Частота', fontsize=8)
            plt.grid(axis='y', alpha=0.75)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.get_current_fig_manager().set_window_title('Гистограммы распределения признаков')
        plt.savefig(PLOTS_DIR + 'гистограммы_признаков.png')
        plt.show()
        plt.close()


        # Матрица корреляции
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Матрица корреляции', fontsize=16)
        plt.get_current_fig_manager().set_window_title('Матрица корреляции')
        plt.savefig(PLOTS_DIR + 'матрица_корреляции.png')
        plt.show()
        plt.close()


    def prepare_data(self, test_size=0.2, random_state=42):
        '''
        Разделяет данные на обучающую и тестовую выборки и масштабирует признаки.
        '''
        if self.data is None:
            print('Ошибка: Данные не загружены. Сначала загрузите данные.')
            return

        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Масштабирование данных
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print('\nДанные разделены на обучающую и тестовую выборки и масштабированы.')


    def train_and_evaluate_model(self, model, model_name, params=None):
        '''
        Обучает и оценивает регрессионную модель.
        '''
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            print('Ошибка: Данные не подготовлены. Сначала подготовьте данные.')
            return None

        print(f'Обучение модели: {model_name}')

        if params:
            grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            print(f'Лучшие параметры для {model_name}: {grid_search.best_params_}')
        else:
            best_model = model.fit(self.X_train, self.y_train)

        y_pred = best_model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f'Результаты для {model_name}:')
        print(f'Среднеквадратичная ошибка (MSE): {mse:.4f}')
        print(f'Коэффициент детерминации (R^2): {r2:.4f}')

        # Визуализация результатов
        self.visualize_predictions(self.y_test, y_pred, model_name)

        return {'model_name': model_name, 'mse': mse, 'r2': r2}


    def visualize_predictions(self, y_true, y_pred, model_name):
        '''
        Визуализирует предсказанные и фактические значения.
        '''
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, color='skyblue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Идеальное предсказание')
        plt.xlabel('Фактические значения', fontsize=12)
        plt.ylabel('Предсказанные значения', fontsize=12)
        plt.title(f'Фактические vs Предсказанные значения ({model_name})', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.get_current_fig_manager().set_window_title(f'Фактические vs Предсказанные значения ({model_name})')
        plt.savefig(PLOTS_DIR + f'predictions_{model_name}.png')
        plt.show()
        plt.close()


    def compare_models(self, results):
        '''
        Сравнивает результаты нескольких моделей.
        '''
        if not results:
            print('Нет результатов для сравнения.')
            return

        df_results = pd.DataFrame(results)
        df_results = df_results.set_index('model_name')

        print('\nСравнение моделей:')
        print(df_results)

        # Визуализация сравнения
        self.visualize_model_comparison(df_results)


    def visualize_model_comparison(self, df_results):
        '''
        Визуализирует сравнение моделей с использованием столбчатой диаграммы.
        '''
        model_names = df_results.index.tolist()
        mse_values = df_results['mse'].tolist()
        r2_values = df_results['r2'].tolist()

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.canvas.manager.set_window_title('Сравнение моделей')

        # График для MSE (слева)
        rects1 = ax1.bar(x - width/2, mse_values, width, label='MSE', color='skyblue')
        ax1.set_ylabel('Среднеквадратичная ошибка (MSE)', color='skyblue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Создаем второй график для R^2 (справа)
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, r2_values, width, label='R^2', color='lightcoral')
        ax2.set_ylabel('Коэффициент детерминации (R^2)', color='lightcoral', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='lightcoral')

        # Настройка графика
        ax1.set_title('Сравнение производительности моделей', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax1.set_xlabel('Модели', fontsize=12)

        # Добавление легенды для обоих графиков
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

        # Подписи значений над столбцами
        def autolabel(rects, ax, color):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Смещение по вертикали
                            textcoords='offset points',
                            ha='center', va='bottom', color=color)

        autolabel(rects1, ax1, 'navy')
        autolabel(rects2, ax2, 'firebrick')

        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(PLOTS_DIR + 'сравнение_моделей.png')
        plt.show()
        plt.close()


def main():
    '''
    Основная функция для запуска обучения и оценки моделей.
    '''
    warnings.filterwarnings("ignore")
    
    data_path = 'diabetes.csv'
    target_column = 'Outcome'
    trainer = RegressionModelTrainer(data_path, target_column)

    # Загрузка и подготовка данных
    trainer.load_data()
    trainer.visualize_data()
    trainer.prepare_data()

    # Обучение и оценка базовой модели
    linear_regression = LinearRegression()
    linear_regression_results = trainer.train_and_evaluate_model(
        linear_regression, 'Линейная регрессия'
    )

    # Обучение и оценка модели с L1 регуляризацией (Lasso)
    lasso = Lasso(max_iter=10000)
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_results = trainer.train_and_evaluate_model(
        lasso, 'Lasso (L1 регуляризация)', params=lasso_params
    )

    # Обучение и оценка модели с L2 регуляризацией (Ridge)
    ridge = Ridge()
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    ridge_results = trainer.train_and_evaluate_model(
        ridge, 'Ridge (L2 регуляризация)', params=ridge_params
    )

    # Сравнение моделей
    results = [linear_regression_results, lasso_results, ridge_results]
    trainer.compare_models(results)


if __name__ == '__main__':
    main()