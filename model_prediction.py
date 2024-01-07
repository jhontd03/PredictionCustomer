"""
Este script contiene funciones para la evaluación y comparación de modelos de aprendizaje automático en un contexto de clasificación. Las funciones proporcionan herramientas detalladas para analizar el rendimiento de los modelos a través de varias métricas, incluyendo precisión, recall, curvas ROC y Precision-Recall, así como gráficos informativos. Estas funciones son esenciales para la evaluación objetiva de modelos en aplicaciones de e-commerce, donde es crucial predecir la conversión de usuarios a clientes de manera precisa.

El script utiliza bibliotecas como pandas para manipulación de datos, matplotlib y seaborn para visualizaciones, y scikit-learn para métricas de evaluación del modelo. Las funciones están diseñadas para ser flexibles y permitir la evaluación detallada de una variedad de modelos de clasificación.

Funciones:
- report: Genera un informe detallado del rendimiento del modelo, incluyendo métricas clave y gráficos.
- model_memory_size: Calcula el tamaño en memoria del modelo.
- confusion_plot: Crea un gráfico de la matriz de confusión.
- feature_importance_plot: Muestra un gráfico de la importancia de las características del modelo.
- roc_plot: Genera la curva ROC para evaluar la capacidad discriminativa del modelo.
- precision_recall_plot: Crea un gráfico de la curva Precision-Recall.
- compare_models: Compara diferentes modelos en base a una métrica específica.

Las funciones son versátiles y pueden ser utilizadas para modelos de clasificación en una variedad de contextos de datos, proporcionando una comprensión clara y precisa del rendimiento del modelo.
"""

# Adaptado de: https://www.kaggle.com/code/para24/survival-prediction-using-cost-sensitive-learning

import sys 
import timeit
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, \
                            precision_recall_curve, roc_curve, accuracy_score

def report(model, model_name, X_train, y_train, X_test, y_test, 
           display_scores=[], importance_plot=False, 
           confusion_labels=None, feature_labels=None, 
           verbose=True):
    """
    Genera un informe completo para evaluar el rendimiento de un modelo de clasificación.

    Parámetros:
    - model: El modelo de clasificación entrenado.
    - model_name: El nombre del modelo (cadena).
    - X_train: Las características de entrenamiento.
    - y_train: Las etiquetas de entrenamiento.
    - X_test: Las características de prueba.
    - y_test: Las etiquetas de prueba.
    - display_scores: Una lista de funciones de puntuación personalizadas para mostrar en el informe (por defecto, []).
    - importance_plot: Un indicador para generar un gráfico de importancia de características (por defecto, False).
    - confusion_labels: Una lista de etiquetas para personalizar la matriz de confusión (por defecto, None).
    - feature_labels: Una lista de etiquetas para las características (por defecto, None).
    - verbose: Un indicador para mostrar información detallada (por defecto, True).

    Salida:
    - Un informe completo que incluye métricas de rendimiento y gráficos.    
    """    
    ## Train
    start = timeit.default_timer()
    train_predictions = model.predict(X_train)
    end = timeit.default_timer()
    train_time = end - start
    
    ## Testing
    start = timeit.default_timer()
    test_predictions = model.predict(X_test)
    end = timeit.default_timer()
    test_time = end - start
    
    train_acc = accuracy_score(y_train, train_predictions)
    test_acc = accuracy_score(y_test, test_predictions)
    
    y_probs = model.predict_proba(X_test)[:, 1]
        
    roc_auc = roc_auc_score(y_test, y_probs)
            
    ## Additional scores
    scores_dict = dict()
    for func in display_scores:
        scores_dict[func.__name__] = [func(y_train, train_predictions),
                                      func(y_test, test_predictions)]
            
    ## Model Memory
    model_mem = round(model_memory_size(model) / 1024, 2)
    
    print(model)
    print("\n==================================> TRAIN-TEST DETAILS <====================================\n")
    
    ## Metrics
    print(f"Train Size: {X_train.shape[0]} samples")
    print(f" Test Size: {X_test.shape[0]} samples")
    print("---------------------------------------------")
    print(f"Training Time: {round(train_time, 3)} seconds")
    print(f" Testing Time: {round(test_time, 3)} seconds")
    print("---------------------------------------------")
    print("Train Accuracy: ", train_acc)
    print(" Test Accuracy: ", test_acc)
    print("---------------------------------------------")
    
    if display_scores:
        for k, v in scores_dict.items():
            score_name = ' '.join(map(lambda x: x.title(), k.split('_')))
            print(f'Train {score_name}: ', v[0])
            print(f' Test {score_name}: ', v[1])
            print()
        print("---------------------------------------------")
    
    print(" Area Under ROC (test): ", roc_auc)
    print("---------------------------------------------")
    print(f"Model Memory Size: {model_mem} kB")
    
    print("\n===============================> CLASSIFICATION REPORT <====================================\n")
    
    ## Classification Report
    model_rep = classification_report(y_test, test_predictions, output_dict=True)
    
    print(classification_report(y_test, test_predictions,
                                target_names=confusion_labels))


    if verbose:
        print("\n================================> CONFUSION MATRIX <=====================================\n")
    
        display(confusion_plot(confusion_matrix(y_test, test_predictions),
                               labels=confusion_labels))
        print("\n=======================================> PLOTS <=========================================\n")
    
    
        ## Variable importance plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        roc_axes = axes[0, 0]
        pr_axes = axes[0, 1]
        importances = None
    
        if importance_plot:
            if not feature_labels:
                raise RuntimeError("'feature_labels' argument not passed "
                                   "when 'importance_plot' is True")
    
            try:
                importances = pd.Series(model.feature_importances_,
                                        index=feature_labels) \
                                .sort_values(ascending=False)
            except AttributeError:
                try:
                    importances = pd.Series(model.coef_.ravel(),
                                            index=feature_labels) \
                                    .sort_values(ascending=False)
                except AttributeError:
                    pass
    
            if importances is not None:
                # Modifying grid
                grid_spec = axes[0, 0].get_gridspec()
                for ax in axes[:, 0]:
                    ax.remove()   # remove first column axes
                large_axs = fig.add_subplot(grid_spec[0:, 0])
    
                # Plot importance curve
                feature_importance_plot(importances=importances.values,
                                        feature_labels=importances.index,
                                        ax=large_axs)
                large_axs.axvline(x=0)
    
                # Axis for ROC and PR curve
                roc_axes = axes[0, 1]
                pr_axes = axes[1, 1]
            else:
                # remove second row axes
                for ax in axes[1, :]:
                    ax.remove()
        else:
            # remove second row axes
            for ax in axes[1, :]:
                ax.remove()
    
    
        ## ROC and Precision-Recall curves
        roc_plot(y_test, y_probs, model_name, ax=roc_axes)
        precision_recall_plot(y_true=y_test, y_probs=y_probs, label=model_name, ax=pr_axes)
    
        fig.subplots_adjust(wspace=5)
        fig.tight_layout()
        
        display(fig)

    dump_return = dict(model=model, model_name=model_name, accuracy=[train_acc, test_acc], **scores_dict,
                       train_time=train_time, train_predictions=train_predictions,
                       test_time=test_time, test_predictions=test_predictions,
                       test_probs=y_probs, report=model_rep, roc_auc=roc_auc,
                       model_memory=model_mem)

    return model, dump_return

def model_memory_size(model):
    """
    Calcula el tamaño en memoria del modelo.

    Parámetros:
    - model: El modelo cuyo tamaño en memoria se va a calcular.

    Salida:
    - El tamaño en memoria del modelo en kilobytes (KB).
    """    
    return sys.getsizeof(pickle.dumps(model))

def confusion_plot(matrix, labels=None):
    """
    Genera una matriz de confusión como un gráfico de calor.

    Parámetros:
    - matrix: La matriz de confusión (pandas DataFrame o matriz NumPy).
    - labels: Una lista de etiquetas para personalizar la matriz de confusión (por defecto, None).

    Salida:
    - Un gráfico de calor que representa la matriz de confusión.
    """    
    labels = labels if labels else ['Negative (0)', 'Positive (1)']
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(data=matrix, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    plt.close()
    
    return fig

def feature_importance_plot(importances, feature_labels, ax=None):
    """
    Genera un gráfico de barras de importancia de características.

    Parámetros:
    - importances: Las importancias de características (valores numéricos).
    - feature_labels: Las etiquetas de características correspondientes.
    - ax: Un objeto de eje opcional para dibujar en un gráfico existente (por defecto, None).

    Salida:
    - Un gráfico de barras que muestra la importancia de las características.
    """    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x=importances, y=feature_labels, ax=axis)
    axis.set_title('Feature Importance Measures')
    
    plt.close()
    
    return axis if ax else fig

def roc_plot(y_true, y_probs, label, compare=False, ax=None):
    """
    Genera un gráfico de la curva ROC.

    Parámetros:
    - y_true: Las etiquetas verdaderas.
    - y_probs: Las probabilidades pronosticadas.
    - label: La etiqueta del modelo.
    - compare: Un indicador para comparar varios modelos en el mismo gráfico (por defecto, False).
    - ax: Un objeto de eje opcional para dibujar en un gráfico existente (por defecto, None).

    Salida:
    - Un gráfico de la curva ROC.
    """    
    fpr, tpr, thresh = roc_curve(y_true, y_probs,
                                 drop_intermediate=False)
    auc = round(roc_auc_score(y_true, y_probs), 2)
    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    label = ' '.join([label, f'({auc})']) if compare else None
    sns.lineplot(x=fpr, y=tpr, ax=axis, label=label)
    
    if compare:
        axis.legend(title='Classifier (AUC)', loc='lower right')
    else:
        axis.text(0.72, 0.05, f'AUC = { auc }', fontsize=12,
                  bbox=dict(facecolor='green', alpha=0.4, pad=5))
            
        # Plot No-Info classifier
        axis.fill_between(fpr, fpr, tpr, alpha=0.3, edgecolor='g',
                          linestyle='--', linewidth=2)
        
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('ROC Curve')
    axis.set_xlabel('False Positive Rate [FPR]\n(1 - Specificity)')
    axis.set_ylabel('True Positive Rate [TPR]\n(Sensitivity or Recall)')
    
    plt.close()
    
    return axis if ax else fig


def precision_recall_plot(y_true, y_probs, label, compare=False, ax=None):
    """
    Genera un gráfico de la curva Precision-Recall.

    Parámetros:
    - y_true: Las etiquetas verdaderas.
    - y_probs: Las probabilidades pronosticadas.
    - label: La etiqueta del modelo.
    - compare: Un indicador para comparar varios modelos en el mismo gráfico (por defecto, False).
    - ax: Un objeto de eje opcional para dibujar en un gráfico existente (por defecto, None).

    Salida:
    - Un gráfico de la curva Precision-Recall.
    """
    p, r, thresh = precision_recall_curve(y_true, y_probs)
    p, r, thresh = list(p), list(r), list(thresh)
    p.pop()  # Remove the last element to match the lengths
    r.pop()  # Remove the last element to match the lengths
    
    # Create a figure and axis if ax is not provided
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    
    # Plot comparing different classifiers
    if compare:
        sns.lineplot(x=r, y=p, ax=axis, label=label)  # Use named arguments here
        axis.set_xlabel('Recall')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')
    else:
        # Plot precision vs threshold
        sns.lineplot(x=thresh, y=p, ax=axis, label='Precision')  # Use named arguments here
        axis.set_xlabel('Threshold')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')

        # Create a twin y-axis to plot recall vs threshold
        axis_twin = axis.twinx()
        sns.lineplot(x=thresh, y=r, ax=axis_twin, color='limegreen', label='Recall')  # Use named arguments here
        axis_twin.set_ylabel('Recall')
        axis_twin.set_ylim(0, 1)
        axis_twin.legend(bbox_to_anchor=(0.24, 0.18))
    
    # Set limits and title
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Precision Vs Recall')
    
    # If the figure was created in this function, close it to avoid double display
    if not ax:
        plt.close(fig)
    
    # Return the appropriate object depending on what was provided
    return axis if ax else fig

def compare_models(clf_reports=[], labels=[], score='accuracy'):
    """
    Compara los informes de varios modelos y crea una tabla de comparación.

    Parámetros:
    - clf_reports: Una lista de informes de modelos generados por la función 'report'.
    - labels: Las etiquetas de los modelos (por defecto, []).
    - score: La métrica de rendimiento para mostrar en la tabla (por defecto, 'accuracy').

    Salida:
    - Una tabla que compara los informes de los modelos.
    """
    ## Classifier Labels
    clf_names = [rep['model_name'] for rep in clf_reports]
        
    ## Compare Table
    table = dict()
    index = ['Train ' + score, 'Test ' + score, 'Overfitting', 'ROC Area',
             'Precision', 'Recall', 'F1-score', 'Support']
    for i in range(len(clf_reports)):
        scores = [round(i, 3) for i in clf_reports[i][score]]
        
        roc_auc = clf_reports[i]['roc_auc']
        
        # Get metrics of True Positive class from sklearn classification_report
        true_positive_metrics = list(clf_reports[i]['report']["1"].values())
        
        table[clf_names[i]] = scores + [scores[1] < scores[0], roc_auc] + \
                              true_positive_metrics
    
    table = pd.DataFrame(data=table, index=index)
    
    return table.T
