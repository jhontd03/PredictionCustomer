import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

def string_split(s):
    return s.replace(',', '')

def change_names_cols(consumos, empresas, primer_consumo, sesiones, usuarios, ventas):
    """
    Cambia los nombres de las columnas a minúsculas y renombra algunas columnas específicas en un conjunto de DataFrames.

    Parámetros:
    - consumos (pandas.DataFrame): Un DataFrame con nombres de columnas a cambiar a minúsculas.
    - empresas (pandas.DataFrame): Un DataFrame con nombres de columnas a cambiar a minúsculas.
    - primer_consumo (pandas.DataFrame): Un DataFrame con nombres de columnas a cambiar a minúsculas.
    - sesiones (pandas.DataFrame): Un DataFrame con nombres de columnas a cambiar a minúsculas.
    - usuarios (pandas.DataFrame): Un DataFrame con nombres de columnas a cambiar a minúsculas.
    - ventas (pandas.DataFrame): Un DataFrame con nombres de columnas a cambiar a minúsculas y algunas columnas a renombrar.

    Salida:
    - Seis DataFrames con los nombres de sus columnas cambiados a minúsculas y algunas columnas renombradas en el DataFrame    
    """
    consumos.columns = [word.lower() for word in consumos.columns]
    empresas.columns = [word.lower() for word in empresas.columns]
    primer_consumo.columns = [word.lower() for word in primer_consumo.columns]
    sesiones.columns = [word.lower() for word in sesiones.columns]
    usuarios.columns = [word.lower() for word in usuarios.columns]
    ventas.columns = [word.lower() for word in ventas.columns]
    ventas = ventas.rename(columns={'vp informe': 'vp_informe', 'vp listado' : 'vp_listado', 'id_usuario': 'idusuario'})
    return consumos, empresas, primer_consumo, sesiones, usuarios, ventas

def value_counts(consumos, empresas, primer_consumo, sesiones, usuarios, ventas):
    """
    Calcula el conteo de tipos de datos en cada DataFrame proporcionado.

    Parámetros:
    - consumos (pandas.DataFrame): Un DataFrame para calcular el conteo de tipos de datos.
    - empresas (pandas.DataFrame): Un DataFrame para calcular el conteo de tipos de datos.
    - primer_consumo (pandas.DataFrame): Un DataFrame para calcular el conteo de tipos de datos.
    - sesiones (pandas.DataFrame): Un DataFrame para calcular el conteo de tipos de datos.
    - usuarios (pandas.DataFrame): Un DataFrame para calcular el conteo de tipos de datos.
    - ventas (pandas.DataFrame): Un DataFrame para calcular el conteo de tipos de datos.

    Salida:
    - Un DataFrame que muestra el conteo de tipos de datos en cada uno de los DataFrames proporcionados.    
    """
    dtypes_df = dict()
    dtypes_df['consumos'] = consumos.dtypes.value_counts()
    dtypes_df['empresas'] = empresas.dtypes.value_counts()
    dtypes_df['primer_consumo'] = primer_consumo.dtypes.value_counts()
    dtypes_df['sesiones'] = sesiones.dtypes.value_counts()
    dtypes_df['usuarios'] = usuarios.dtypes.value_counts()
    dtypes_df['ventas'] = ventas.dtypes.value_counts()
    dtypes_df = pd.DataFrame(dtypes_df).fillna(0)
    return dtypes_df

def dtypes_feat(consumos, empresas, primer_consumo, sesiones, usuarios, ventas):
    """
    Obtiene un DataFrame que muestra los tipos de datos de las características en cada DataFrame proporcionado.

    Parámetros:
    - consumos (pandas.DataFrame): Un DataFrame para obtener los tipos de datos de sus características.
    - empresas (pandas.DataFrame): Un DataFrame para obtener los tipos de datos de sus características.
    - primer_consumo (pandas.DataFrame): Un DataFrame para obtener los tipos de datos de sus características.
    - sesiones (pandas.DataFrame): Un DataFrame para obtener los tipos de datos de sus características.
    - usuarios (pandas.DataFrame): Un DataFrame para obtener los tipos de datos de sus características.
    - ventas (pandas.DataFrame): Un DataFrame para obtener los tipos de datos de sus características.

    Salida:
    - Un DataFrame que muestra los tipos de datos de las características en cada uno de los DataFrames proporcionados.    
    """
    dtypes_feat = dict()
    dtypes_feat['consumos'] = consumos.dtypes.to_frame().T
    dtypes_feat['empresas'] = empresas.dtypes.to_frame().T
    dtypes_feat['primer_consumo'] = primer_consumo.dtypes.to_frame().T
    dtypes_feat['sesiones'] = sesiones.dtypes.to_frame().T
    dtypes_feat['usuarios'] = usuarios.dtypes.to_frame().T
    dtypes_feat['ventas'] = ventas.dtypes.to_frame().T
    dtypes_feat = pd.concat([dtypes_feat[item_type].rename(index={0:item_type}) for item_type in dtypes_feat]).T.fillna(0)
    return dtypes_feat

def normalize_df(consumos, empresas, primer_consumo, sesiones, usuarios, ventas):
    """
    Realiza la normalización de filas en DataFrames específicos.

    Parámetros:
    - consumos (pandas.DataFrame): Un DataFrame que se normalizará.
    - empresas (pandas.DataFrame): Un DataFrame que se normalizará.
    - primer_consumo (pandas.DataFrame): Un DataFrame que se normalizará.
    - sesiones (pandas.DataFrame): Un DataFrame que se normalizará.
    - usuarios (pandas.DataFrame): Un DataFrame que se normalizará.
    - ventas (pandas.DataFrame): Un DataFrame que se normalizará.

    Salida:
    - Seis DataFrames con filas normalizadas de acuerdo a las operaciones realizadas.
    """
    # Normalización de filas en consumos
    consumos.update(consumos.loc[:, ['idconsumo', 'idusuario', 'idproducto', 'empconsul_id']].applymap(string_split))
    consumos['empconsul_prov_desc'] = consumos['empconsul_prov_desc'].str.capitalize()
    consumos['fechaconsumo'] = pd.to_datetime(consumos['fechaconsumo'], format='%d/%m/%Y %H:%M:%S')
    
    # Normalización de filas en empresas
    empresas.update(empresas.loc[:, ['empconsul_id']].applymap(string_split))
    empresas['empconsul_prov_desc'] = empresas['empconsul_prov_desc'].str.capitalize()
    
    # Normalización de filas en primer_consumo
    primer_consumo.update(primer_consumo.loc[:, ['idconsumo', 'idusuario', 'idproducto', 'empconsul_id']].applymap(string_split))
    primer_consumo['empconsul_prov_desc'] = primer_consumo['empconsul_prov_desc'].str.capitalize()
    primer_consumo['fechaconsumo'] = pd.to_datetime(primer_consumo['fechaconsumo'], format='%d/%m/%Y %H:%M:%S')
    
    # Normalización de filas en sesiones
    sesiones.update(sesiones.loc[:, ['idusuario']].applymap(string_split))
    sesiones['fecha_sesion'] = pd.to_datetime(sesiones['fecha_sesion'], format='%d/%m/%Y %H:%M:%S')
    
    # Normalización de filas en usuarios
    usuarios.update(usuarios.loc[:, ['idusuario']].applymap(string_split))
    usuarios['fec_registro'] = pd.to_datetime(usuarios['fec_registro'], format='%d/%m/%Y %H:%M:%S')
    usuarios['fec_cliente'] = pd.to_datetime(usuarios['fec_cliente'], format='%d/%m/%Y %H:%M:%S')
    usuarios['ind_cliente'] = usuarios['ind_cliente'].astype('object')
    usuarios['canal_registro'] = usuarios['canal_registro'].astype('object')
    usuarios['bondad_email'] = usuarios['bondad_email'].astype('object')
    
    # Normalización de filas en ventas
    ventas.update(ventas.loc[:, ['idventa', 'idusuario', 'importes']].applymap(string_split))
    ventas['importes'] = ventas['importes'].astype('float')
    ventas['importe'] = ventas['importe'].astype('float')
    ventas['numventas'] = ventas['numventas'].astype('int')
    ventas['fechaventa'] = pd.to_datetime(ventas['fechaventa'], format='%d/%m/%Y %H:%M:%S')
        
    return consumos, empresas, primer_consumo, sesiones, usuarios, ventas

def null_value_counts(consumos, empresas, primer_consumo, sesiones, usuarios, ventas):
    """
    Calcula el conteo de valores nulos en cada columna de los DataFrames proporcionados.

    Parámetros:
    - consumos (pandas.DataFrame): Un DataFrame para calcular el conteo de valores nulos en sus columnas.
    - empresas (pandas.DataFrame): Un DataFrame para calcular el conteo de valores nulos en sus columnas.
    - primer_consumo (pandas.DataFrame): Un DataFrame para calcular el conteo de valores nulos en sus columnas.
    - sesiones (pandas.DataFrame): Un DataFrame para calcular el conteo de valores nulos en sus columnas.
    - usuarios (pandas.DataFrame): Un DataFrame para calcular el conteo de valores nulos en sus columnas.
    - ventas (pandas.DataFrame): Un DataFrame para calcular el conteo de valores nulos en sus columnas.

    Salida:
    - Un DataFrame que muestra el conteo de valores nulos en cada columna de los DataFrames proporcionados.
    """
    dtypes_null_df = dict()
    dtypes_null_df['consumos'] = consumos.isnull().sum().sort_values(ascending=False)
    dtypes_null_df['empresas'] = empresas.isnull().sum().sort_values(ascending=False)
    dtypes_null_df['primer_consumo'] = primer_consumo.isnull().sum().sort_values(ascending=False)
    dtypes_null_df['sesiones'] = sesiones.isnull().sum().sort_values(ascending=False)
    dtypes_null_df['usuarios'] = usuarios.isnull().sum().sort_values(ascending=False)
    dtypes_null_df['ventas'] = ventas.isnull().sum().sort_values(ascending=False)
    dtypes_null_df = pd.DataFrame(dtypes_null_df).fillna(0)
    return dtypes_null_df


def most_common(series):
    """
    Encuentra el valor más común en una serie, ignorando los valores nulos.

    Parámetros:
    - series (pandas.Series): Una serie de datos de la cual se desea encontrar el valor más común.

    Salida:
    - El valor más común en la serie, o None si la serie no contiene valores no nulos.    
    """
    valid_series = series.dropna()  
    if not valid_series.empty:
        return valid_series.value_counts().idxmax()
    return None

def most_common_count(series):
    """
    Calcula la cantidad de valores únicos más comunes en una serie, ignorando los valores nulos.

    Parámetros:
    - series (pandas.Series): Una serie de datos de la cual se desea calcular la cantidad de valores únicos más comunes.

    Salida:
    - El número de valores únicos más comunes en la serie, o None si la serie no contiene valores no nulos.    
    """
    valid_series = series.dropna()  
    if not valid_series.empty:
        return len(valid_series.value_counts())
    return None

def process_partition_mc(partition):
    """
    Procesa una partición de datos y calcula estadísticas agregadas por usuario.

    Parámetros:
    - partition (pandas.DataFrame): Un DataFrame que contiene datos de una partición.

    Salida:
    - Un DataFrame con estadísticas agregadas por usuario, incluyendo la fecha mínima y máxima de consumo, 
      el producto más común, la empresa consultada más común, el sector más común y la provincia más común,
      la suma del número de consumos y la suma del número de usuarios.    
    """
    return partition.groupby('idusuario').agg({
        'fechaconsumo': ['min', 'max'],
        'descproducto': most_common,    
        'empconsul_id': most_common,
        'empconsul_sector': most_common,
        'empconsul_prov_desc': most_common,    
        'numconsumos': 'sum',
        'numusuarios': 'sum'
    }).reset_index()

def process_partition_mcc(partition):
    """
    Procesa una partición de datos y calcula la cantidad de valores únicos más comunes por usuario.

    Parámetros:
    - partition (pandas.DataFrame): Un DataFrame que contiene datos de una partición.

    Salida:
    - Un DataFrame con la cantidad de valores únicos más comunes por usuario para las columnas especificadas.
    """
    return partition.groupby('idusuario').agg({
        'descproducto': most_common_count,
        'empconsul_id': most_common_count,
        'empconsul_sector': most_common_count,
        'empconsul_prov_desc': most_common_count,
    }).reset_index()

def extract_info_consumos(data, n_partitions, n_cpus=6):
    """
    Procesa una partición de datos y calcula la cantidad de valores únicos más comunes por usuario.

    Parámetros:
    - partition (pandas.DataFrame): Un DataFrame que contiene datos de una partición.

    Salida:
    - Un DataFrame con la cantidad de valores únicos más comunes por usuario para las columnas especificadas.
    """
    data_partitions = np.array_split(data, n_partitions)
    
    results_mc = Parallel(n_jobs=n_cpus, backend="threading")(delayed(process_partition_mc)(partition) for partition in data_partitions)
    consumos_df1 = pd.concat(results_mc, axis=0).reset_index(drop=True)
    consumos_df1.columns = ['idusuario', 'fecha_primera_consulta', 'fecha_ultima_consulta',
                            'descproducto_mas_freq', 'empconsul_id_mas_freq', 'empconsul_sector_mas_freq',
                            'empconsul_prov_desc_mas_freq', 'total_consumos', 'total_usuarios']
    
    results_mcc = Parallel(n_jobs=n_cpus, backend="threading")(delayed(process_partition_mcc)(partition) for partition in data_partitions)
    consumos_df2 = pd.concat(results_mcc, axis=0).reset_index(drop=True)
    consumos_df2.columns = ['idusuario', 'descproducto_mas_freq_total', 
                            'empconsul_id_mas_freq_total', 'empconsul_sector_mas_freq_total',
                            'empconsul_prov_desc_mas_freq_total']

    consumos_df1.set_index('idusuario', inplace=True)
    consumos_df2.set_index('idusuario', inplace=True)
    consumos_df3 = pd.concat([consumos_df1, consumos_df2], axis=1)

    return consumos_df3

def info_sesiones(sesion):
    """
    Calcula información relevante sobre las sesiones de un usuario.

    Parámetros:
    - sesion (tuple): Una tupla que contiene el ID de usuario y un DataFrame de sesiones asociado.

    Salida:
    - Un diccionario que contiene información relevante sobre las sesiones del usuario.
    """    
    item_sesion = sesion[1].sort_values(by='fecha_sesion')
    info_sesion = dict()
    info_sesion['fecha_inicio_sesion'] = item_sesion['fecha_sesion'].iloc[0].strftime('%Y-%m-%d')
    info_sesion['fecha_fin_sesion'] = item_sesion['fecha_sesion'].iloc[-1].strftime('%Y-%m-%d')
    if len(item_sesion) > 1:
        days_diff = item_sesion['fecha_sesion'].diff()
        info_sesion['dias_max_entre_sesion'] = days_diff.max().days
        info_sesion['dias_min_entre_sesion'] = days_diff.min().days                
    else:
        info_sesion['dias_max_entre_sesion'] = 0
        info_sesion['dias_min_entre_sesion'] = 0
    info_sesion['fechas_total_sesion'] = len(item_sesion)
    info_sesion['total_sesiones'] = item_sesion['sesiones'].sum()
    return {sesion[0] : info_sesion}

def cols_impute(data, cols_to_impute, mode='mode', value=None):
    """
    Imputa valores faltantes en columnas específicas de un DataFrame.

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene los datos.
    - cols_to_impute (list): Una lista de nombres de columnas en las que se imputarán los valores faltantes.
    - mode (str): El método de imputación a utilizar. Puede ser 'mode' (por defecto) para imputar con la moda,
      'mean' para imputar con la media, o 'value' para imputar con un valor específico proporcionado.
    - value: El valor específico a utilizar para la imputación cuando mode='value'. Ignorado si mode es diferente de 'value'.

    Salida:
    - Un nuevo DataFrame con los valores faltantes imputados en las columnas especificadas.
    """    
    data_impute = data.copy(deep=True)
    if mode == 'mode':
        imputation_num_dict = data_impute[cols_to_impute].mode().iloc[0].to_dict()
    elif mode == 'mean':
        imputation_num_dict = data_impute[cols_to_impute].mean().iloc[0].to_dict()    
    else:
        imputation_num_dict = {col: value for col in cols_to_impute}
    data_impute.fillna(imputation_num_dict, inplace=True)
    return data_impute

def compare_hist(data_original, data_impute):
    """
    Compara las distribuciones de dos conjuntos de datos y muestra un gráfico de densidad.

    Parámetros:
    - data_original (pandas.Series): Una serie de datos original.
    - data_impute (pandas.Series): Una serie de datos imputados.

    Salida:
    - Un gráfico de densidad que compara las distribuciones de los datos original e imputado.
    """    
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data_original.dropna(), fill=True, color='red', label='Original')
    sns.kdeplot(data_impute, fill=True, color='green', label='Imputado')
    plt.title('Comparación de Distribuciones')
    plt.xlabel(data_original.name)
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

def impute_num_pred(data, cols_to_impute, n_estimators=10, add_flag=False):
    """
    Imputa valores numéricos faltantes utilizando un modelo de regresión iterativa (IterativeImputer).

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene los datos.
    - cols_to_impute (list): Una lista de nombres de columnas numéricas en las que se imputarán los valores faltantes.
    - n_estimators (int): El número de estimadores (árboles) a utilizar en el modelo de regresión (por defecto, 10).
    - add_flag (bool): Un indicador para agregar columnas de bandera (flag) que indican si los valores fueron imputados (por defecto, False).

    Salida:
    - Un nuevo DataFrame con los valores numéricos faltantes imputados en las columnas especificadas.
    """    
    data_original = data.copy(deep=True)
    imputer = IterativeImputer(random_state=0, estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=42))
    data_impute = imputer.fit_transform(data_original[cols_to_impute])     
    data_impute = [pd.Series([i[index_col] for i in data_impute], name=name_col)
        for index_col, name_col in enumerate(cols_to_impute)]   
    data_impute = pd.concat(data_impute, axis=1)
    index_name = data_original.index.name
    data_original.reset_index(inplace=True)
    data_original.update(data_impute)
    data_original.set_index(index_name, inplace=True)
    if add_flag:
        columns_flag = generate_flag_null(data, cols_to_impute)
        return pd.concat([data_original, columns_flag], axis=1)
    else:
        return data_original

def impute_cat_pred(data, cols_to_impute, n_estimators=10, add_flag=False):
    """
    Imputa valores numéricos faltantes utilizando un modelo de regresión iterativa (IterativeImputer).

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene los datos.
    - cols_to_impute (list): Una lista de nombres de columnas numéricas en las que se imputarán los valores faltantes.
    - n_estimators (int): El número de estimadores (árboles) a utilizar en el modelo de regresión (por defecto, 10).
    - add_flag (bool): Un indicador para agregar columnas de bandera (flag) que indican si los valores fueron imputados (por defecto, False).

    Salida:
    - Un nuevo DataFrame con los valores numéricos faltantes imputados en las columnas especificadas.
    """
    data_original = data.copy(deep=True)
    
    impute_df = list()
    for col_name in cols_to_impute:
        encode_col = LabelEncoder()
        data_original['encode_col'] = encode_col.fit_transform(data_original[col_name].astype(str))
        imputer = IterativeImputer(random_state=0, estimator=RandomForestClassifier(n_estimators=n_estimators, random_state=42))
        data_impute = imputer.fit_transform(data_original[['encode_col']])
        data_impute = pd.Series([i[0] for i in data_impute], name='encode_col')
        impute_df.append(pd.Series(encode_col.inverse_transform(data_impute.astype(int)), name=col_name))
        
    data_impute = pd.concat(impute_df, axis=1)    
    index_name = data.index.name
    data.reset_index(inplace=True)
    data[cols_to_impute] = data_impute
    data.set_index(index_name, inplace=True)

    if add_flag:
        columns_flag = generate_flag_null(data, cols_to_impute)
        return pd.concat([data, columns_flag], axis=1)
    else:
        return data

def generate_flag_null(data, cols_to_impute):
    """
    Genera columnas de bandera (flag) que indican la presencia de valores nulos en las columnas especificadas.

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene los datos.
    - cols_to_impute (list): Una lista de nombres de columnas en las que se buscarán valores nulos.

    Salida:
    - Un nuevo DataFrame que incluye columnas de bandera para cada columna especificada en 'cols_to_impute'.
    """    
    data_flag = data.copy(deep=True)
    for col in data:
        data_flag[f'{col}_flag_null'] = data_flag[col].isnull().astype(int)
    return data_flag

def extract_cols_impute(data, mode='num'):
    """
    Extrae las columnas que contienen valores nulos para su imputación.

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene los datos.
    - mode (str): El modo de extracción de columnas, 'num' para columnas numéricas o 'cat' para columnas categóricas (por defecto, 'num').

    Salida:
    - Una lista de nombres de columnas que contienen valores nulos y son elegibles para imputación.
    """    
    if mode == 'num':
        cols_to_impute = data.select_dtypes(include='float').isnull().mean()
    else:
        cols_to_impute = data.select_dtypes(include='object').isnull().mean()        
    return list(cols_to_impute[cols_to_impute > 0].index)


def plot_boxplot_and_hist(data):
    """
    Genera gráficos de caja y histogramas para las variables numéricas en un DataFrame.

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene las variables numéricas a visualizar.

    Salida:
    - Gráficos de caja y histogramas para cada variable numérica en el DataFrame.
    """    
    variables = list(data.columns)
    n_vars = len(variables)
    n_rows = (n_vars + 1) // 2
    n_cols = 2 if n_vars > 1 else 1
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(16, 8 * n_rows), 
                              gridspec_kw={"height_ratios": [0.8, 1] * n_rows})

    if n_vars == 1:
        axes = axes.reshape(2, -1)

    # Desactivamos todos los ejes para luego activar solo los necesarios
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    for i, var in enumerate(variables):
        row_idx = (i // n_cols) * 2
        col_idx = i % n_cols

        axes[row_idx, col_idx].axis('on')
        axes[row_idx + 1, col_idx].axis('on')
        
        sns.boxplot(x=data[var], ax=axes[row_idx, col_idx])
        axes[row_idx, col_idx].set(xlabel="")
        axes[row_idx, col_idx].set(title=var)

        axes[row_idx + 1, col_idx].hist(data[var], bins=30, edgecolor="k", alpha=0.7)
        axes[row_idx + 1, col_idx].set(title=var)

    plt.tight_layout()
    plt.show()

    
def plot_categorical_bars(data, show_counts=False):
    """
    Genera gráficos de barras para las variables categóricas en un DataFrame.

    Parámetros:
    - data (pandas.DataFrame): Un DataFrame que contiene las variables categóricas a visualizar.
    - show_counts (bool): Un indicador para mostrar los conteos en las barras (por defecto, False).

    Salida:
    - Gráficos de barras para cada variable categórica en el DataFrame.
    """   
    variables = list(data.columns)
    n_vars = len(variables)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    if n_vars % 2 == 1:
        fig.delaxes(axes.flatten()[-1])

    for ax, var in zip(axes.flatten(), variables):

        counts = data[var].value_counts()
        
        bars = ax.bar(counts.index, counts.values)
        
        ax.set_title(var)
        ax.set_ylabel('Count')
        ax.set_xticks(counts.index)
        ax.set_xticklabels(counts.index, rotation=90)

        # Si show_counts es True, añadir label con el número de conteo en cada barra
        if show_counts:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 5, 
                        str(int(height)), ha='center', va='bottom', 
                        rotation=90, fontsize=8)

    plt.tight_layout()
    plt.show()


def describe(df):
    """
    Calcula estadísticos descriptivos para cada columna de un DataFrame de Pandas.

    Esta función calcula varias medidas estadísticas comunes, que incluyen:
    - count (conteo): Número de elementos no nulos.
    - mean (media): Valor medio de los elementos.
    - std (desviación estándar): Medida de la dispersión de los elementos.
    - min (mínimo): Valor mínimo de los elementos.
    - 25% (percentil 25): Valor debajo del cual se encuentran el 25% de los elementos.
    - 50% (percentil 50 o mediana): Valor medio de los elementos.
    - 75% (percentil 75): Valor debajo del cual se encuentran el 75% de los elementos.
    - max (máximo): Valor máximo de los elementos.
    - skew (asimetría): Medida de la asimetría de la distribución de los elementos.
    - kurtosis (curtosis): Medida de la "agudeza" de la distribución de los elementos.

    Parámetros:
    df (pandas.DataFrame): DataFrame del cual se calcularán los estadísticos.

    Devuelve:
    pandas.DataFrame: Un DataFrame que contiene los estadísticos descriptivos calculados
                      para cada columna del DataFrame de entrada.
    """
    funciones = {
        'count': 'count',
        'mean': 'mean',
        'std': 'std',
        'min': 'min',
        '25%': lambda x: x.quantile(0.25),
        '50%': lambda x: x.quantile(0.5),
        '75%': lambda x: x.quantile(0.75),
        'max': 'max',
        'skew': 'skew',
        'kurtosis': 'kurt'
    }  
    descripcion_agg = df.agg(funciones)
    return descripcion_agg      